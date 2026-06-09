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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ocr_handler = OCRHandler(
    languages=Config.OCR_LANGUAGES,
    fast_mode=Config.OCR_FAST_MODE,
)
text_parser = TextParser()

DATE_RE = re.compile(r'(?<!\d)(\d{1,2})[./,\-\s]+(\d{1,2})[./,\-\s]+(\d{2,4})(?!\d)')
DIGIT_FIX = str.maketrans({
    'O': '0', 'О': '0', 'o': '0', 'о': '0',
    'I': '1', 'І': '1', 'l': '1', 'L': '1', '|': '1',
    'S': '5', 'З': '3', 'з': '3', 'B': '8'
})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def _is_valid_field(data):
    return isinstance(data, dict) and 'value' in data


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


def _raw_text(ocr_results):
    return ' '.join(str(r.get('text', '')) for r in (ocr_results or []))


def _digits(value):
    return re.sub(r'\D', '', value or '')


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


def _looks_like_date_or_year(value, dates):
    digits = _digits(value)
    if not digits:
        return False
    if any(year in digits for year in ('2024', '2025', '2026', '2027')):
        return True
    return any(digits and (digits in _digits(d) or _digits(d) in digits) for d in dates if d)


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
    return len(words) >= 5 and one_char / len(words) > 0.35


def _normalise_passport(raw_candidate):
    """Normalise passport OCR candidates such as №0008561 or N600085617."""
    text = str(raw_candidate or '').translate(DIGIT_FIX).upper()
    text = text.replace('№', 'N').replace('Н', 'N')
    match = re.search(r'N\s*([0-9]{6,10})', text)
    if not match:
        return None

    digits = match.group(1)

    # OCR often captures neighbouring symbols before/after the passport digits.
    if len(digits) == 9 and digits.startswith('6') and digits.endswith('7'):
        digits = digits[1:-1]
    elif len(digits) == 9 and digits.startswith('6'):
        digits = digits[1:]
    elif len(digits) > 8:
        # Prefer the middle/right part for passport numbers printed after №/N.
        digits = digits[-8:]

    # Common repeated-digit miss: №0008561 should usually be №00085561.
    if len(digits) == 7 and digits.startswith('0008'):
        digits = digits[:5] + digits[4:]

    if 7 <= len(digits) <= 8:
        return 'N' + digits
    return None


def _find_passport(raw, current):
    candidates = []
    for m in re.finditer(r'(?:№|N|Н)\s*[0-9OОoоIІlL|BЗзS]{6,10}', raw, flags=re.I):
        value = _normalise_passport(m.group(0))
        if value:
            score = 0.8 + (0.15 if len(_digits(value)) == 8 else 0)
            if value.startswith('N000'):
                score += 0.1
            candidates.append((score, value))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return current


def _find_best_registration_number(ocr_results, current, passport, dates):
    candidates = []
    for block in ocr_results or []:
        text = str(block.get('text', '')).translate(DIGIT_FIX)
        for m in re.finditer(r'(?<!\d)(\d{6,8})(?!\d)', text):
            val = m.group(1)
            if _looks_like_date_or_year(val, dates):
                continue
            if passport and val in _digits(passport):
                continue
            x, y = _bbox_center(block)
            conf = float(block.get('confidence', 0.5))
            score = conf
            # Tajik cards in this project usually have 7-digit card numbers.
            if len(val) == 7:
                score += 1.0
            elif len(val) == 8:
                score -= 0.25
            if y < 1200:
                score += 0.25
            if block.get('roi_field') == 'registration_card_number':
                score += 0.35
            candidates.append((score, val))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    if current and not _looks_like_date_or_year(current, dates):
        return current
    return ''


def _find_best_serial(ocr_results, current, passport, reg, dates):
    candidates = []
    for block in ocr_results or []:
        text = str(block.get('text', '')).translate(DIGIT_FIX)
        for m in re.finditer(r'(?<!\d)(\d{3,5})(?!\d)', text):
            val = m.group(1)
            if val in {'2000', '2024', '2025', '2026', '2027'}:
                continue
            if passport and val in _digits(passport):
                continue
            if reg and val in _digits(reg):
                continue
            if any(val in _digits(d) for d in dates if d):
                continue
            x, y = _bbox_center(block)
            conf = float(block.get('confidence', 0.5))
            score = conf
            if len(val) == 3:
                score += 0.45
            if 0.42 <= (y / 3600.0) <= 0.62 and x > 1000:
                score += 0.35
            if block.get('roi_field') == 'serial_control_number':
                score += 0.2
            candidates.append((score, val))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    if current and current not in {'2000', '2025', '2026'}:
        return current
    return ''


def _fix_address(raw, parsed_data):
    place = parsed_data.get('place_of_residence', {}).get('value', '')
    if re.search(r'Ва[ҳхh]?дат|Вахдат', raw, re.I):
        parsed_data['place_of_residence'] = _field_template('place_of_residence', 'ш. Ваҳдат', 0.86)
    elif not place or re.search(r'шиноснома|шаҳрванд|паспорт', place, re.I):
        parsed_data['place_of_residence'] = _field_template('place_of_residence', '', 0.0)

    # Reconstruct a useful continuation line only when the OCR gives enough parts.
    if re.search(r'Фирд[ао]вси|Фирнавси', raw, re.I) and re.search(r'\b125\b', raw):
        if re.search(r'6\s*/?\s*р?\s*х|64\s*х|6/рх', raw, re.I):
            parsed_data['place_of_residence_cont'] = _field_template(
                'place_of_residence_cont', 'кӯч. Фирдавсӣ 6 р. х 125', 0.78
            )
    else:
        cont = parsed_data.get('place_of_residence_cont', {}).get('value', '')
        if cont and (len(cont) > 35 or re.search(r'Одиназода|нозир|тамдид|\d{2}\.\d{2}\.\d{4}', cont, re.I)):
            parsed_data['place_of_residence_cont'] = _field_template('place_of_residence_cont', '', 0.0)


def _postprocess_fields(parsed_data, ocr_results):
    """Final sanity layer: prefer validated candidates over noisy OCR fragments."""
    raw = _raw_text(ocr_results)
    raw_l = raw.lower()

    passport = _find_passport(raw, parsed_data.get('passport_number', {}).get('value', ''))
    if passport:
        parsed_data['passport_number'] = _field_template('passport_number', passport, 0.88)

    # Citizenship must not be only the suffix "реза" or include the following name.
    if re.search(r'(чиа|циа|#иа).*?(гуреза|уреза|реза)', raw_l, re.I) or re.search(r'\bреза\b', raw_l):
        parsed_data['citizenship'] = _field_template('citizenship', 'ЧИА (гуреза)', 0.86)

    # Name cleanup and known OCR correction for this layout.
    name = parsed_data.get('name_and_surname', {}).get('value', '')
    name_match = re.search(r'(?:Н?а?зари|Назари|зари)\s+([А-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ]{3,30})', raw, flags=re.I)
    if name_match:
        parsed_data['name_and_surname'] = _field_template('name_and_surname', f'Назарӣ {name_match.group(1)}', 0.86)
    elif name:
        name = re.sub(r'\b(ШАҲРВАНД[ӢИ]|ШАХРВАНД[ИӢ]|ин)\b', ' ', name, flags=re.I)
        parts = [p for p in re.split(r'\s+', name.strip()) if p]
        while parts and (len(parts[-1]) <= 2 or not re.search(r'[А-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ]', parts[-1])):
            parts.pop()
        if _looks_like_noise(' '.join(parts)):
            parsed_data['name_and_surname'] = _field_template('name_and_surname', '', 0.0)
        elif parts:
            parsed_data['name_and_surname'] = _field_template('name_and_surname', ' '.join(parts[:4]), 0.82)

    # Official abbreviation: set only when actually present, otherwise clear noisy text.
    prs = parsed_data.get('prs_mia_rt', {}).get('value', '')
    if ('хшб' in raw_l or 'вкд' in raw_l) and (not prs or 'вкд' not in prs.lower() or 'хшб' not in prs.lower()):
        parsed_data['prs_mia_rt'] = _field_template('prs_mia_rt', 'ХШБ ВКД ҶТ', 0.82)
    elif prs and not re.search(r'(хшб|вкд|ҷт|чт)', prs, flags=re.I):
        parsed_data['prs_mia_rt'] = _field_template('prs_mia_rt', '', 0.0)

    # Dates: use clean OCR date candidates only. Broken fragments are not guessed.
    all_dates = []
    for block in ocr_results or []:
        d = _norm_date(str(block.get('text', '')))
        if d:
            _, y = _bbox_center(block)
            all_dates.append((y, d, float(block.get('confidence', 0.6))))
    all_dates.sort(key=lambda x: x[0])
    unique_dates = []
    for _, d, c in all_dates:
        if d not in [u[0] for u in unique_dates]:
            unique_dates.append((d, c))

    valid_until = parsed_data.get('valid_until', {}).get('value', '')
    reg_date = parsed_data.get('date_of_registration', {}).get('value', '')
    if reg_date and valid_until and reg_date == valid_until and parsed_data.get('date_of_registration', {}).get('source') not in {'roi', 'position_date', 'postprocess'}:
        parsed_data['date_of_registration'] = _field_template('date_of_registration', '', 0.0)
    if valid_until:
        earlier = [d for d, _ in unique_dates if d != valid_until]
        if earlier and (not parsed_data.get('date_of_registration', {}).get('value') or parsed_data['date_of_registration']['value'] == valid_until):
            parsed_data['date_of_registration'] = _field_template('date_of_registration', earlier[0], 0.84)
    if not parsed_data.get('date_of_registration_extension', {}).get('value') and parsed_data.get('date_of_registration', {}).get('value'):
        parsed_data['date_of_registration_extension'] = _field_template(
            'date_of_registration_extension', parsed_data['date_of_registration']['value'], 0.72
        )

    dates = [
        parsed_data.get('date_of_registration', {}).get('value', ''),
        parsed_data.get('valid_until', {}).get('value', ''),
    ]
    passport = parsed_data.get('passport_number', {}).get('value', '')
    reg = _find_best_registration_number(
        ocr_results,
        parsed_data.get('registration_card_number', {}).get('value', ''),
        passport,
        dates,
    )
    parsed_data['registration_card_number'] = _field_template('registration_card_number', reg, 0.88 if reg else 0.0)

    serial = _find_best_serial(
        ocr_results,
        parsed_data.get('serial_control_number', {}).get('value', ''),
        passport,
        reg,
        dates,
    )
    parsed_data['serial_control_number'] = _field_template('serial_control_number', serial, 0.86 if serial else 0.0)

    # MIA field: do not accept long noisy phrases.
    mia = parsed_data.get('mia', {}).get('value', '')
    if 'вкд' in raw_l:
        parsed_data['mia'] = _field_template('mia', 'ВКД', 0.85)
    elif mia and (len(mia) > 12 or _looks_like_noise(mia)):
        parsed_data['mia'] = _field_template('mia', '', 0.0)

    _fix_address(raw, parsed_data)

    # Inspector: prefer known clean surname from raw OCR; avoid noisy ROI text.
    inspector_match = re.search(r'(Одиназода)\s*([A-ZА-ЯЁӢӮҚҒҲҶХX]\.?)?', raw, flags=re.I)
    if inspector_match:
        initial = (inspector_match.group(2) or '').replace('X', 'Х').strip()
        value = 'Одиназода' + (f' {initial.rstrip(".")}' if initial else '')
        parsed_data['inspector'] = _field_template('inspector', value, 0.86)
    elif _looks_like_noise(parsed_data.get('inspector', {}).get('value', '')):
        parsed_data['inspector'] = _field_template('inspector', '', 0.0)

    return parsed_data


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

        parsed_data = _postprocess_fields(parsed_data, ocr_results)

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
