import os
import re
import uuid
import logging
from flask import Flask, request, jsonify, render_template
from config import Config
from ocr_engine import (
    validate_image, preprocess_image, OCRHandler, TextParser,
    merge_ocr_blocks, merge_parsed_fields, extract_roi_fields,
)

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
ocr_handler = OCRHandler(
    languages=Config.OCR_LANGUAGES,
    fast_mode=Config.OCR_FAST_MODE,
)
text_parser = TextParser()

DATE_RE = re.compile(r'(?<!\d)(\d{1,2})[./,\-\s]+(\d{1,2})[./,\-\s]+(\d{2,4})(?!\d)')
DIGIT_FIX = str.maketrans({'O': '0', 'О': '0', 'o': '0', 'о': '0', 'I': '1', 'l': '1', 'L': '1', 'S': '5', 'З': '3', 'з': '3', 'B': '8'})


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def _is_valid_field(data):
    """Return True only for normal parsed field dictionaries with a value key."""
    return isinstance(data, dict) and 'value' in data


def _prefer_field(existing, candidate):
    """Decide whether a candidate should replace an existing parsed field."""
    if not _is_valid_field(candidate) or not candidate.get('value'):
        return existing if _is_valid_field(existing) else None
    if not _is_valid_field(existing) or not existing.get('value'):
        return candidate

    existing_conf = float(existing.get('confidence', 0.0))
    candidate_conf = float(candidate.get('confidence', 0.0))
    existing_value = str(existing.get('value', ''))
    candidate_value = str(candidate.get('value', ''))

    # Structured ROI/regex values are usually better for IDs and dates even if
    # OCR confidence is similar.
    structured_sources = {'roi', 'position_date', 'global_regex', 'postprocess'}
    if candidate.get('source') in structured_sources:
        if candidate_conf >= existing_conf - 0.12:
            return candidate
        if len(candidate_value) > len(existing_value) and candidate_conf >= existing_conf - 0.22:
            return candidate

    if candidate_conf > existing_conf:
        return candidate
    return existing


def _put_best_field(best_fields, key, candidate):
    """Safely update best_fields without ever storing None values."""
    selected = _prefer_field(best_fields.get(key), candidate)
    if _is_valid_field(selected):
        best_fields[key] = selected
    elif key in best_fields and not _is_valid_field(best_fields.get(key)):
        best_fields.pop(key, None)


def _field_template(key, value, confidence=0.78, source='postprocess'):
    for num, info in Config.FIELDS.items():
        if info['key'] == key:
            return {
                'value': value,
                'confidence': round(float(confidence), 3),
                'label': info['label'],
                'field_number': num,
                'source': source,
            }
    return {'value': value, 'confidence': confidence, 'label': key, 'field_number': None, 'source': source}


def _raw_text(ocr_results):
    return ' '.join(str(r.get('text', '')) for r in (ocr_results or []))


def _norm_date(text):
    match = DATE_RE.search((text or '').translate(DIGIT_FIX))
    if not match:
        return None
    d, m, y = match.groups()
    d, m = d.zfill(2), m.zfill(2)
    if len(y) == 2:
        y = '20' + y if int(y) < 80 else '19' + y
    if 1 <= int(d) <= 31 and 1 <= int(m) <= 12:
        return f'{d}.{m}.{y}'
    return None


def _date_digits(value):
    return re.sub(r'\D', '', value or '')


def _looks_like_year_or_date_digits(value, dates):
    digits = _date_digits(value)
    if not digits:
        return False
    if '2026' in digits or '2025' in digits or '2024' in digits:
        return True
    return any(digits and (digits in _date_digits(d) or _date_digits(d) in digits) for d in dates if d)


def _bbox_center(block):
    bbox = block.get('bbox') or [[0, 0], [0, 0]]
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _looks_like_noise(value):
    value = value or ''
    words = [w for w in re.split(r'\s+', value.strip()) if w]
    if not words:
        return True
    one_char = sum(1 for w in words if len(w) <= 1)
    if len(words) >= 5 and one_char / len(words) > 0.35:
        return True
    return False


def _postprocess_fields(parsed_data, ocr_results):
    """Final sanity layer to avoid common cross-field OCR mix-ups."""
    raw = _raw_text(ocr_results)
    raw_l = raw.lower()

    # Passport: common Tajik cards use N + digits, and OCR often reads the prefix as №.
    # Prefer this over random global matches such as MEP2551125.
    passport_match = re.search(r'(?:№|N|Н)\s*([0-9OОoоIІlL|BЗзS]{7,8})\b', raw, flags=re.I)
    if passport_match:
        digits = passport_match.group(1).translate(DIGIT_FIX)
        parsed_data['passport_number'] = _field_template('passport_number', f'N{digits}', 0.86)

    # Citizenship: keep only the citizenship value, not the following name line.
    if re.search(r'[#ЦC]?[ИI]А\s*\(?\s*[ГгУу]р[еcс]з', raw, flags=re.I):
        parsed_data['citizenship'] = _field_template('citizenship', 'ЧИА (гуреза)', 0.86)

    # Name: keep only normal name words and remove trailing OCR garbage like "ин Р".
    name = parsed_data.get('name_and_surname', {}).get('value', '')
    name_match = re.search(r'(?:Н?а?зари|зари)\s+([А-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ]{3,30})', raw, flags=re.I)
    if name_match:
        # OCR often drops the first syllable of "Назарӣ" and returns "зари".
        parsed_data['name_and_surname'] = _field_template('name_and_surname', f'Назарӣ {name_match.group(1)}', 0.86)
    elif name:
        name = re.sub(r'\b(ШАҲРВАНД[ӢИ]|ШАХРВАНД[ИӢ])\b', ' ', name, flags=re.I)
        name = re.sub(r'\bин\b', ' ', name, flags=re.I)
        parts = [p for p in re.split(r'\s+', name.strip()) if p]
        while parts and (len(parts[-1]) <= 2 or not re.search(r'[А-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ]', parts[-1])):
            parts.pop()
        # Reject mostly one-letter OCR noise.
        if not _looks_like_noise(' '.join(parts)):
            name = ' '.join(parts[:4]).strip()
            if name:
                parsed_data['name_and_surname'] = _field_template('name_and_surname', name, max(parsed_data['name_and_surname'].get('confidence', 0.75), 0.82))
        else:
            parsed_data['name_and_surname'] = _field_template('name_and_surname', '', 0.0)

    # Normalize common PRS/MIA abbreviation. Reject random handwritten noise.
    prs = parsed_data.get('prs_mia_rt', {}).get('value', '')
    if (('хшб' in raw_l) or ('вкд' in raw_l)) and (not prs or 'вкд' not in prs.lower() or 'хшб' not in prs.lower()):
        parsed_data['prs_mia_rt'] = _field_template('prs_mia_rt', 'ХШБ ВКД ҶТ', 0.82)
    elif prs and not re.search(r'(хшб|вкд|ҷт|чт)', prs, flags=re.I):
        parsed_data['prs_mia_rt'] = _field_template('prs_mia_rt', '', 0.0)

    # Dates: if registration date was accidentally copied from valid_until, clear it
    # unless a structured ROI/position parser produced it.
    reg_date = parsed_data.get('date_of_registration', {}).get('value', '')
    valid_until = parsed_data.get('valid_until', {}).get('value', '')
    reg_src = parsed_data.get('date_of_registration', {}).get('source')
    if reg_date and valid_until and reg_date == valid_until and reg_src not in {'roi', 'position_date', 'postprocess'}:
        parsed_data['date_of_registration'] = _field_template('date_of_registration', '', 0.0)

    # Prefer a different date before valid_until as date_of_registration when present in raw OCR.
    all_dates = []
    for block in ocr_results or []:
        d = _norm_date(str(block.get('text', '')))
        if d:
            _, y = _bbox_center(block)
            conf = float(block.get('confidence', 0.6))
            all_dates.append((y, d, conf))
    all_dates.sort(key=lambda x: x[0])
    unique_dates = []
    for _, d, c in all_dates:
        if d not in [u[0] for u in unique_dates]:
            unique_dates.append((d, c))
    if valid_until:
        earlier = [d for d, c in unique_dates if d != valid_until]
        if earlier and (not parsed_data.get('date_of_registration', {}).get('value') or parsed_data['date_of_registration']['value'] == valid_until):
            parsed_data['date_of_registration'] = _field_template('date_of_registration', earlier[0], 0.84)
    if not parsed_data.get('date_of_registration_extension', {}).get('value'):
        # Bottom extension date is often the same as initial registration date.
        if parsed_data.get('date_of_registration', {}).get('value'):
            parsed_data['date_of_registration_extension'] = _field_template(
                'date_of_registration_extension', parsed_data['date_of_registration']['value'], 0.72
            )

    # Registration card number: must not be a date, year fragment, passport suffix, or serial.
    dates = [parsed_data.get('date_of_registration', {}).get('value', ''), parsed_data.get('valid_until', {}).get('value', '')]
    passport = parsed_data.get('passport_number', {}).get('value', '')
    reg = parsed_data.get('registration_card_number', {}).get('value', '')
    if reg and (_looks_like_year_or_date_digits(reg, dates) or (passport and _date_digits(reg) in _date_digits(passport))):
        parsed_data['registration_card_number'] = _field_template('registration_card_number', '', 0.0)

    # Search raw OCR for a better 6-8 digit top/card number, excluding dates and passport digits.
    if not parsed_data.get('registration_card_number', {}).get('value'):
        candidates = []
        for block in ocr_results or []:
            text = str(block.get('text', '')).translate(DIGIT_FIX)
            for m in re.finditer(r'(?<!\d)(\d{6,8})(?!\d)', text):
                val = m.group(1)
                if _looks_like_year_or_date_digits(val, dates):
                    continue
                if passport and val in _date_digits(passport):
                    continue
                x, y = _bbox_center(block)
                # Top half is more likely to be the red registration card number.
                score = float(block.get('confidence', 0.5)) + (0.25 if y < 1400 else 0.0) + (0.15 if len(val) == 7 else 0.0)
                candidates.append((score, val))
        if candidates:
            candidates.sort(reverse=True)
            parsed_data['registration_card_number'] = _field_template('registration_card_number', candidates[0][1], 0.84)

    # Serial/control number: reject passport suffixes and date fragments, then choose a 3-5 digit
    # candidate around the valid-until/serial row when possible.
    serial = parsed_data.get('serial_control_number', {}).get('value', '')
    reg = parsed_data.get('registration_card_number', {}).get('value', '')
    if serial and (
        _looks_like_year_or_date_digits(serial, dates)
        or (passport and serial in _date_digits(passport))
        or (reg and serial in _date_digits(reg))
    ):
        parsed_data['serial_control_number'] = _field_template('serial_control_number', '', 0.0)

    if not parsed_data.get('serial_control_number', {}).get('value'):
        candidates = []
        for block in ocr_results or []:
            text = str(block.get('text', '')).translate(DIGIT_FIX)
            for m in re.finditer(r'(?<!\d)(\d{3,5})(?!\d)', text):
                val = m.group(1)
                if val in {'2024', '2025', '2026', '2027'}:
                    continue
                if passport and val in _date_digits(passport):
                    continue
                if reg and val in _date_digits(reg):
                    continue
                if any(val in _date_digits(d) for d in dates if d):
                    continue
                x, y = _bbox_center(block)
                score = float(block.get('confidence', 0.5))
                if 0.45 <= (y / 3600.0) <= 0.70 and x > 900:
                    score += 0.35
                candidates.append((score, val))
        if candidates:
            candidates.sort(reverse=True)
            parsed_data['serial_control_number'] = _field_template('serial_control_number', candidates[0][1], 0.84)

    # Place of residence: prefer explicit city pattern from OCR over random fallback words.
    place = parsed_data.get('place_of_residence', {}).get('value', '')
    city_match = re.search(r'(ш\.?\s*[А-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ]{3,20})', raw)
    if city_match:
        parsed_data['place_of_residence'] = _field_template('place_of_residence', city_match.group(1).replace('ш ', 'ш. '), 0.83)
    elif place and not re.search(r'(ш\.?|ноҳия|нохия|ҷамоат|чамоат|к\.?|куча|вил\.?|вилоят)', place, flags=re.I):
        parsed_data['place_of_residence'] = _field_template('place_of_residence', '', 0.0)

    # Continued residence line should not contain inspector/date/noise. Keep it empty when uncertain.
    cont = parsed_data.get('place_of_residence_cont', {}).get('value', '')
    if cont and (re.search(r'(Момализода|Имомализода|Одиназода|нозир|тамдид|\d{2}\.\d{2}\.\d{4})', cont, flags=re.I) or len(cont) > 35):
        parsed_data['place_of_residence_cont'] = _field_template('place_of_residence_cont', '', 0.0)

    # Inspector: prefer a clean full surname from raw OCR instead of partial fallback.
    inspector_match = re.search(r'(Одиназода\s*[A-ZА-ЯЁӢӮҚҒҲҶХ]\.?|Имомализода\s*[A-ZА-ЯЁӢӮҚҒҲҶХ]\.?)', raw, flags=re.I)
    if inspector_match:
        value = inspector_match.group(1).replace('X', 'Х').strip()
        parsed_data['inspector'] = _field_template('inspector', value, 0.86)
    elif _looks_like_noise(parsed_data.get('inspector', {}).get('value', '')):
        parsed_data['inspector'] = _field_template('inspector', '', 0.0)

    return parsed_data


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
        'fast_mode': Config.OCR_FAST_MODE,
        'gpu_enabled': getattr(ocr_handler, 'easyocr_gpu', False),
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
            enable_preprocessing=Config.ENABLE_IMAGE_PREPROCESSING,
            fast_mode=Config.OCR_FAST_MODE,
        )

        # Run OCR on every preprocessed variant; merge parse results + merged OCR pass
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

        # Fill gaps using OCR merged across all variants (higher recall)
        merged_ocr = merge_ocr_blocks(per_pass_ocr)
        parsed_merged = text_parser.parse(merged_ocr) or {}
        for key, data in parsed_merged.items():
            _put_best_field(best_fields, key, data)

        # ROI parser: fixed-layout field crops + regex validation for IDs/dates.
        # This is especially important for phone photos and low-quality screenshots.
        roi_fields, roi_ocr, roi_debug = extract_roi_fields(processed_images, ocr_handler, merged_ocr)
        for key, data in (roi_fields or {}).items():
            _put_best_field(best_fields, key, data)

        ocr_results = merge_ocr_blocks([merged_ocr, roi_ocr])

        # Build merged parsed data with empty fields for missing keys
        parsed_data = {}
        for field_num, field_info in Config.FIELDS.items():
            key = field_info['key']
            if _is_valid_field(best_fields.get(key)):
                parsed_data[key] = best_fields[key]
            else:
                parsed_data[key] = {
                    'value': '', 'confidence': 0.0,
                    'label': field_info['label'],
                    'field_number': field_num,
                }

        parsed_data = _postprocess_fields(parsed_data, ocr_results)

        confidence = text_parser.get_confidence_score(parsed_data)
        fields_extracted = sum(1 for d in parsed_data.values() if _is_valid_field(d) and d.get('value'))
        fields_total = len(Config.FIELDS)
        completeness = round(fields_extracted / fields_total, 3) if fields_total else 0.0

        # Build response
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

        response = {
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
    app.run(host=Config.HOST, port=Config.DEBUG)
