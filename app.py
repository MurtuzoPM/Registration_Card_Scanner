import os
import uuid
import logging
from flask import Flask, request, jsonify, render_template
from config import Config
from ocr_engine import (
    validate_image, preprocess_image, create_ocr_handler, TextParser,
    merge_ocr_blocks, merge_parsed_fields, extract_roi_fields,
    postprocess_fields,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ocr_handler = create_ocr_handler(
    languages=Config.OCR_LANGUAGES,
    fast_mode=Config.OCR_FAST_MODE,
)
text_parser = TextParser()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def _is_valid_field(data):
    return isinstance(data, dict) and 'value' in data


def _prefer_field(existing, candidate):
    if not _is_valid_field(candidate) or not candidate.get('value'):
        return existing if _is_valid_field(existing) else None
    if not _is_valid_field(existing) or not existing.get('value'):
        return candidate

    existing_conf = float(existing.get('confidence', 0.0))
    candidate_conf = float(candidate.get('confidence', 0.0))
    structured_sources = {'roi', 'position_date', 'global_regex', 'postprocess'}
    if candidate.get('source') in structured_sources and candidate_conf >= existing_conf - 0.12:
        return candidate
    if candidate_conf > existing_conf:
        return candidate
    return existing


def _put_best_field(best_fields, key, candidate):
    selected = _prefer_field(best_fields.get(key), candidate)
    if _is_valid_field(selected):
        best_fields[key] = selected
    elif key in best_fields and not _is_valid_field(best_fields.get(key)):
        best_fields.pop(key, None)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'ocr_engine': Config.OCR_ENGINE,
        'languages': Config.OCR_LANGUAGES,
        'fast_mode': Config.OCR_FAST_MODE,
        'gpu_enabled': getattr(ocr_handler, 'easyocr_gpu', False),
        'fields_count': len(Config.FIELDS),
        'fields': {str(k): v['label'] for k, v in Config.FIELDS.items()},
    })


@app.route('/api/extract', methods=['POST'])
def extract():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {Config.ALLOWED_EXTENSIONS}'}), 400

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > Config.MAX_FILE_SIZE:
        return jsonify({'error': f'File too large. Max: {Config.MAX_FILE_SIZE / 1024 / 1024:.0f} MB'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f'{uuid.uuid4().hex}.{ext}'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(file_path)

    try:
        is_valid, validation_msg = validate_image(file_path)
        if not is_valid:
            return jsonify({'error': validation_msg}), 400

        logger.info('Processing image: %s', file_path)
        processed_images = preprocess_image(
            file_path,
            enable_preprocessing=Config.ENABLE_IMAGE_PREPROCESSING,
            fast_mode=Config.OCR_FAST_MODE,
        )

        per_pass_ocr = []
        per_pass_parsed = []
        for img in processed_images:
            ocr_results = ocr_handler.extract_text_with_positions(img)
            per_pass_ocr.append(ocr_results)
            parsed = text_parser.parse(ocr_results) or {}
            per_pass_parsed.append({k: v for k, v in parsed.items() if _is_valid_field(v)})

        best_fields = {
            k: v for k, v in (merge_parsed_fields(per_pass_parsed) or {}).items()
            if _is_valid_field(v)
        }

        merged_ocr = merge_ocr_blocks(per_pass_ocr)
        parsed_merged = text_parser.parse(merged_ocr) or {}
        for key, data in parsed_merged.items():
            _put_best_field(best_fields, key, data)

        roi_fields, roi_ocr, roi_debug = extract_roi_fields(processed_images, ocr_handler, merged_ocr)
        for key, data in (roi_fields or {}).items():
            _put_best_field(best_fields, key, data)

        ocr_results = merge_ocr_blocks([merged_ocr, roi_ocr])

        parsed_data = {}
        for field_num, field_info in Config.FIELDS.items():
            key = field_info['key']
            if _is_valid_field(best_fields.get(key)):
                parsed_data[key] = best_fields[key]
            else:
                parsed_data[key] = {
                    'value': '',
                    'confidence': 0.0,
                    'label': field_info['label'],
                    'field_number': field_num,
                }

        parsed_data = postprocess_fields(parsed_data, ocr_results)

        confidence = text_parser.get_confidence_score(parsed_data)
        fields_extracted = sum(1 for d in parsed_data.values() if _is_valid_field(d) and d.get('value'))
        fields_total = len(Config.FIELDS)
        completeness = round(fields_extracted / fields_total, 3) if fields_total else 0.0

        fields = {}
        for key, data in parsed_data.items():
            fields[key] = {
                'value': data['value'],
                'confidence': data['confidence'],
                'label': data['label'],
                'field_number': data['field_number'],
            }
            if data.get('source'):
                fields[key]['source'] = data['source']

        return jsonify({
            'success': True,
            'fields': fields,
            'confidence': confidence,
            'fields_extracted': fields_extracted,
            'fields_total': fields_total,
            'completeness': completeness,
            'debug': roi_debug,
            'raw_ocr': [
                {'text': r['text'], 'confidence': r['confidence'], 'bbox': r['bbox']}
                for r in ocr_results
            ],
        })

    except Exception as e:
        logger.error('Processing error: %s', e, exc_info=True)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning('Failed to remove temp file %s: %s', file_path, e)


if __name__ == '__main__':
    logger.info('Starting Registration Card Scanner on port %s', Config.PORT)
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
