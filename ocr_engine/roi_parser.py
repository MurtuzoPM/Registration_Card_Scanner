"""ROI-based extraction for Tajik registration cards.

The printed card layout is stable, so parsing fixed regions of interest (ROIs)
is more reliable than parsing one large OCR text stream.  This module runs OCR on
expected field zones, validates candidates by field type, and returns normal
field dictionaries that can be merged with the existing TextParser output.
"""

import logging
import re
from typing import Dict, Iterable, List, Optional, Tuple

from config import Config
from .tajik_normalize import normalize_text

logger = logging.getLogger(__name__)

BBox = List[List[int]]
OCRBlock = Dict[str, object]
FieldResult = Dict[str, object]
ROI = Tuple[float, float, float, float]


# Relative coordinates after preprocessing.  They intentionally include a little
# extra space around the value area because scanned/phone photos are often skewed.
FIELD_ROIS: Dict[str, List[ROI]] = {
    'registration_card_number': [(0.52, 0.08, 0.96, 0.20), (0.45, 0.10, 0.98, 0.25)],
    'passport_number': [(0.38, 0.21, 0.93, 0.31)],
    'citizenship': [(0.38, 0.28, 0.93, 0.38)],
    'name_and_surname': [(0.28, 0.35, 0.95, 0.46)],
    'date_of_registration': [(0.28, 0.43, 0.78, 0.53)],
    'prs_mia_rt': [(0.28, 0.48, 0.94, 0.59)],
    'valid_until': [(0.18, 0.54, 0.68, 0.64)],
    'serial_control_number': [(0.60, 0.54, 0.95, 0.64)],
    'mia': [(0.42, 0.58, 0.94, 0.69)],
    'place_of_residence': [(0.25, 0.64, 0.95, 0.75)],
    'place_of_residence_cont': [(0.25, 0.70, 0.95, 0.82)],
    'inspector': [(0.25, 0.77, 0.96, 0.89)],
    'date_of_registration_extension': [(0.28, 0.85, 0.86, 0.98)],
}

FIELD_NUMBERS = {info['key']: num for num, info in Config.FIELDS.items()}
FIELD_LABELS = {info['key']: info['label'] for info in Config.FIELDS.values()}

DATE_RE = re.compile(r'(?<!\d)(\d{1,2})[./,\-\s]+(\d{1,2})[./,\-\s]+(\d{2,4})(?!\d)')
PASSPORT_RE = re.compile(r'\b([A-ZА-ЯЁ]{1,3})\s*([A-ZА-ЯЁСC])?\s*(\d[\dOОoоIІlL|BЗЗS]{5,8})\b', re.I)
REG_CARD_RE = re.compile(r'(?<!\d)(\d{6,10})(?!\d)')
SERIAL_RE = re.compile(r'(?<!\d)(\d{4,6})(?!\d)')

LABEL_WORD_RE = re.compile(
    r'\b(ШАҲРВАНД[ӢИ]|ШАХРВАНД[ИӢ]|НОМУ?|НАСАБ|РАҚАМИ|РИҚАМИ|ШИНОСНОМА|'
    r'САНАИ|БАҚАЙДГИР[ӢИ]|БАКАЙДГИР[ИӢ]|ҶОИ|ЧОИ|ЗИСТ|МӮҲЛАТ|МУХЛАТ|'
    r'ЭЪТИБОР|СИЛСИЛА|НАЗОРАТ|НОЗИР|ТАМДИД|ВАЗОРАТИ|КОРҲОИ|ДОХИЛӢ)\b',
    re.I,
)

TEXT_FIXES = {
    'ХШБ ВКД ЧТ': 'ХШБ ВКД ҶТ',
    'ХШБ ВКД ҶТ': 'ХШБ ВКД ҶТ',
    'ВКД ЧТ': 'ВКД ҶТ',
    'Афгонистон': 'Афғонистон',
    'Афганистон': 'Афғонистон',
}

PASSPORT_LETTER_FIXES = str.maketrans({
    'А': 'A', 'а': 'A',
    'В': 'B', 'в': 'B',
    'Е': 'E', 'е': 'E',
    'К': 'K', 'к': 'K',
    'М': 'M', 'м': 'M',
    'Н': 'H', 'н': 'H',
    'О': 'O', 'о': 'O',
    'Р': 'P', 'р': 'P',
    'С': 'C', 'с': 'C',
    'Т': 'T', 'т': 'T',
    'Х': 'X', 'х': 'X',
})

DIGIT_FIXES = str.maketrans({
    'O': '0', 'О': '0', 'o': '0', 'о': '0',
    'I': '1', 'І': '1', 'l': '1', 'L': '1', '|': '1',
    'B': '8', 'S': '5', 'З': '3', 'з': '3',
})


def _field_info(key: str, value: str, confidence: float, source: str = 'roi') -> FieldResult:
    return {
        'value': value,
        'confidence': round(float(confidence), 3),
        'label': FIELD_LABELS.get(key, key),
        'field_number': FIELD_NUMBERS.get(key),
        'source': source,
    }


def _bbox_center_y(block: OCRBlock) -> float:
    bbox = block.get('bbox') or [[0, 0]]
    ys = [p[1] for p in bbox]
    return (min(ys) + max(ys)) / 2


def _bbox_to_global(bbox: BBox, x_offset: int, y_offset: int) -> BBox:
    return [[int(x + x_offset), int(y + y_offset)] for x, y in bbox]


def _crop(img, roi: ROI):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi
    ix1 = max(0, int(x1 * w))
    iy1 = max(0, int(y1 * h))
    ix2 = min(w, int(x2 * w))
    iy2 = min(h, int(y2 * h))
    if ix2 <= ix1 or iy2 <= iy1:
        return None, (0, 0)
    return img[iy1:iy2, ix1:ix2].copy(), (ix1, iy1)


def _join_blocks(blocks: Iterable[OCRBlock]) -> str:
    ordered = sorted(blocks, key=lambda b: (b.get('bbox', [[0, 0]])[0][1], b.get('bbox', [[0, 0]])[0][0]))
    return ' '.join(str(b.get('text', '')).strip() for b in ordered if str(b.get('text', '')).strip())


def _avg_conf(blocks: Iterable[OCRBlock]) -> float:
    vals = [float(b.get('confidence', 0.0)) for b in blocks]
    return sum(vals) / len(vals) if vals else 0.0


def _apply_text_fixes(text: str) -> str:
    text = normalize_text(text or '').strip()
    text = re.sub(r'\s+', ' ', text)
    for old, new in TEXT_FIXES.items():
        text = re.sub(re.escape(old), new, text, flags=re.I)
    return text.strip(' :-–—,;')


def _strip_labels(text: str) -> str:
    text = LABEL_WORD_RE.sub(' ', text)
    text = re.sub(r'\b\d{1,2}\b', ' ', text)
    text = re.sub(r'[():;,_]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip(' -–—')


def _normalize_date(text: str) -> Optional[str]:
    fixed = (text or '').translate(DIGIT_FIXES)
    match = DATE_RE.search(fixed)
    if not match:
        return None
    day, month, year = match.groups()
    day, month = day.zfill(2), month.zfill(2)
    if len(year) == 2:
        year = '20' + year if int(year) < 80 else '19' + year
    try:
        d, m = int(day), int(month)
    except ValueError:
        return None
    if not (1 <= d <= 31 and 1 <= m <= 12):
        return None
    return f'{day}.{month}.{year}'


def _normalize_passport(text: str) -> Optional[str]:
    compact = re.sub(r'[^A-Za-zА-Яа-яЁё0-9]', '', text or '')
    compact = compact.translate(PASSPORT_LETTER_FIXES).upper()
    match = re.search(r'([A-Z]{1,3})([A-Z])?(\d[\dO0I1lLB8S5З3]{5,8})', compact)
    if not match:
        return None
    prefix = match.group(1)
    optional = match.group(2) or ''
    digits = match.group(3).translate(DIGIT_FIXES)
    # Common OCR: P C 06476028, where C is a separator/series artifact.
    if optional in {'C', 'O'} and len(digits) >= 7:
        optional = ''
    candidate = f'{prefix}{optional}{digits}'
    if re.fullmatch(r'[A-Z]{1,3}\d{6,9}', candidate):
        return candidate
    return None


def _find_registration_number(text: str) -> Optional[str]:
    fixed = (text or '').translate(DIGIT_FIXES)
    candidates = [m.group(1) for m in REG_CARD_RE.finditer(fixed)]
    if not candidates:
        return None
    # Registration number is usually 6-8 digits. Prefer 7 digits if present.
    candidates.sort(key=lambda v: (abs(len(v) - 7), -len(v)))
    return candidates[0]


def _find_serial(text: str) -> Optional[str]:
    fixed = (text or '').translate(DIGIT_FIXES)
    candidates = [m.group(1) for m in SERIAL_RE.finditer(fixed)]
    if not candidates:
        return None
    # Avoid picking long document IDs as serials.
    candidates.sort(key=lambda v: (abs(len(v) - 4), len(v)))
    return candidates[0]


def _clean_text_field(key: str, text: str) -> str:
    text = _apply_text_fixes(text)
    text = _strip_labels(text)

    if key == 'name_and_surname':
        # Remove dates/numbers/code fragments from names.
        text = DATE_RE.sub(' ', text)
        text = re.sub(r'\b[A-ZА-ЯЁ]{1,3}\d{5,9}\b', ' ', text, flags=re.I)
        text = re.sub(r'\d+', ' ', text)
    elif key in {'citizenship', 'place_of_residence', 'place_of_residence_cont', 'inspector'}:
        text = DATE_RE.sub(' ', text)
    elif key in {'prs_mia_rt', 'mia'}:
        # Keep short official abbreviations but remove unrelated punctuation.
        text = re.sub(r'[^A-Za-zА-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip(' -–—,.;:')
    return _apply_text_fixes(text)


def _select_value(key: str, blocks: List[OCRBlock]) -> Optional[str]:
    joined = _join_blocks(blocks)
    if not joined:
        return None

    if key == 'registration_card_number':
        return _find_registration_number(joined)
    if key == 'passport_number':
        return _normalize_passport(joined)
    if key in {'date_of_registration', 'valid_until', 'date_of_registration_extension'}:
        return _normalize_date(joined)
    if key == 'serial_control_number':
        return _find_serial(joined)

    cleaned = _clean_text_field(key, joined)
    if len(cleaned) < 2:
        return None
    return cleaned


def _extract_roi_candidates(processed_images, ocr_handler) -> Tuple[Dict[str, FieldResult], List[OCRBlock], Dict[str, object]]:
    fields: Dict[str, FieldResult] = {}
    roi_blocks_all: List[OCRBlock] = []
    debug = {'roi_candidates': {}, 'date_candidates': []}

    for image_index, img in enumerate(processed_images):
        for key, rois in FIELD_ROIS.items():
            for roi_index, roi in enumerate(rois):
                crop, offset = _crop(img, roi)
                if crop is None:
                    continue
                try:
                    blocks = ocr_handler.extract_text_with_positions(crop)
                except Exception as exc:
                    logger.debug('ROI OCR failed for %s image=%d roi=%d: %s', key, image_index, roi_index, exc)
                    continue
                if not blocks:
                    continue

                global_blocks = []
                for block in blocks:
                    gb = dict(block)
                    gb['bbox'] = _bbox_to_global(block.get('bbox', [[0, 0]]), offset[0], offset[1])
                    gb['roi_field'] = key
                    gb['roi_index'] = roi_index
                    gb['image_index'] = image_index
                    global_blocks.append(gb)
                roi_blocks_all.extend(global_blocks)

                raw_text = _join_blocks(blocks)
                selected = _select_value(key, blocks)
                debug['roi_candidates'].setdefault(key, []).append({
                    'raw': raw_text,
                    'selected': selected,
                    'confidence': round(_avg_conf(blocks), 3),
                    'image_index': image_index,
                    'roi_index': roi_index,
                })
                if key in {'date_of_registration', 'valid_until', 'date_of_registration_extension'}:
                    date = _normalize_date(raw_text)
                    if date:
                        debug['date_candidates'].append({'field_hint': key, 'value': date, 'raw': raw_text})

                if not selected:
                    continue

                conf = max(_avg_conf(blocks), 0.65)
                if key in {'registration_card_number', 'passport_number'}:
                    conf = max(conf, 0.88)
                elif key in {'date_of_registration', 'valid_until', 'date_of_registration_extension'}:
                    conf = max(conf, 0.86)
                elif key == 'serial_control_number':
                    conf = max(conf, 0.82)

                current = fields.get(key)
                if current is None or conf > float(current.get('confidence', 0)) or len(selected) > len(str(current.get('value', ''))):
                    fields[key] = _field_info(key, selected, min(conf, 0.99), source='roi')

    return fields, roi_blocks_all, debug


def _assign_dates_from_positions(ocr_blocks: List[OCRBlock], fields: Dict[str, FieldResult]) -> None:
    """Global date fallback: collect all dates and assign by vertical position."""
    dated = []
    max_y = 1
    for block in ocr_blocks:
        bbox = block.get('bbox') or [[0, 0]]
        max_y = max(max_y, max(p[1] for p in bbox))
    for block in ocr_blocks:
        text = str(block.get('text', ''))
        date = _normalize_date(text)
        if date:
            y = _bbox_center_y(block) / max_y
            dated.append((y, date, float(block.get('confidence', 0.7))))
    if not dated:
        return
    dated.sort(key=lambda x: x[0])

    targets = {
        'date_of_registration': (0.34, 0.55),
        'valid_until': (0.48, 0.68),
        'date_of_registration_extension': (0.78, 1.05),
    }
    for key, (low, high) in targets.items():
        if fields.get(key, {}).get('value'):
            continue
        choices = [d for d in dated if low <= d[0] <= high]
        if not choices:
            continue
        # Pick the highest-confidence date in the expected band.
        choices.sort(key=lambda x: x[2], reverse=True)
        fields[key] = _field_info(key, choices[0][1], max(choices[0][2], 0.8), source='position_date')


def _assign_ids_from_global_text(ocr_blocks: List[OCRBlock], fields: Dict[str, FieldResult]) -> None:
    text = ' '.join(str(b.get('text', '')) for b in ocr_blocks)
    if not fields.get('registration_card_number', {}).get('value'):
        reg = _find_registration_number(text)
        if reg:
            fields['registration_card_number'] = _field_info('registration_card_number', reg, 0.82, source='global_regex')
    if not fields.get('passport_number', {}).get('value'):
        passport = _normalize_passport(text)
        if passport:
            fields['passport_number'] = _field_info('passport_number', passport, 0.82, source='global_regex')


def extract_roi_fields(processed_images, ocr_handler, merged_ocr: Optional[List[OCRBlock]] = None):
    """
    Extract fields using fixed ROIs from all processed image variants.

    Returns:
        (fields, roi_ocr_blocks, debug)
    """
    fields, roi_blocks, debug = _extract_roi_candidates(processed_images, ocr_handler)
    all_blocks = list(merged_ocr or []) + roi_blocks
    _assign_dates_from_positions(all_blocks, fields)
    _assign_ids_from_global_text(all_blocks, fields)
    return fields, roi_blocks, debug
