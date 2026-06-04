import os
import uuid
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from config import Config
from ocr_engine import validate_image, preprocess_image, OCRHandler, TextParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OCR engine and parser
ocr_handler = OCRHandler(languages=Config.OCR_LANGUAGES)
text_parser = TextParser()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'ocr_engine': Config.OCR_ENGINE,
        'languages': Config.OCR_LANGUAGES,
        'fields_count': len(Config.FIELDS),
        'fields': {str(k): v['label'] for k, v in Config.FIELDS.items()},
    })


@app.route('/api/extract', methods=['POST'])
def extract():
    """
    Extract registration card data from uploaded image.
    Accepts multipart/form-data with 'image' file field.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {Config.ALLOWED_EXTENSIONS}'}), 400

    # Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > Config.MAX_FILE_SIZE:
        return jsonify({'error': f'File too large. Max: {Config.MAX_FILE_SIZE / 1024 / 1024:.0f} MB'}), 400

    # Save uploaded file
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f'{uuid.uuid4().hex}.{ext}'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(file_path)

    try:
        # Validate image
        is_valid, validation_msg = validate_image(file_path)
        if not is_valid:
            return jsonify({'error': validation_msg}), 400

        # Preprocess image (returns list of images for multi-pass OCR)
        logger.info(f'Processing image: {file_path}')
        processed_images = preprocess_image(
            file_path,
            enable_preprocessing=Config.ENABLE_IMAGE_PREPROCESSING
        )

        # Run OCR + parse on each variant independently, then merge field results
        best_fields = {}
        all_raw_ocr = []
        for img in processed_images:
            ocr_results = ocr_handler.extract_text_with_positions(img)
            all_raw_ocr.extend(ocr_results)
            parsed = text_parser.parse(ocr_results)
            for key, data in parsed.items():
                if data['value'] and (
                    key not in best_fields or
                    data['confidence'] > best_fields[key]['confidence']
                ):
                    best_fields[key] = data

        # Deduplicate raw OCR for response
        seen = {}
        for r in all_raw_ocr:
            k = r['text'].lower().strip()
            if k not in seen or r['confidence'] > seen[k]['confidence']:
                seen[k] = r
        ocr_results = sorted(seen.values(), key=lambda x: (x['bbox'][0][1], x['bbox'][0][0]))

        # Build merged parsed data with empty fields for missing keys
        parsed_data = {}
        for field_num, field_info in Config.FIELDS.items():
            key = field_info['key']
            if key in best_fields:
                parsed_data[key] = best_fields[key]
            else:
                parsed_data[key] = {
                    'value': '', 'confidence': 0.0,
                    'label': field_info['label'],
                    'field_number': field_num,
                }

        confidence = text_parser.get_confidence_score(parsed_data)

        # Build response
        fields = {}
        for key, data in parsed_data.items():
            fields[key] = {
                'value': data['value'],
                'confidence': data['confidence'],
                'label': data['label'],
                'field_number': data['field_number'],
            }

        response = {
            'success': True,
            'fields': fields,
            'confidence': confidence,
            'raw_ocr': [
                {'text': r['text'], 'confidence': r['confidence'], 'bbox': r['bbox']}
                for r in ocr_results
            ],
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f'Processing error: {e}', exc_info=True)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f'Failed to remove temp file {file_path}: {e}')


if __name__ == '__main__':
    logger.info('Starting Registration Card Scanner on port %s', Config.PORT)
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
