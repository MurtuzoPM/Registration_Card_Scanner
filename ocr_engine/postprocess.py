"""Generic, card-agnostic post-processing for registration-card fields.

Single source of truth used by the Flask app and offline tools.

Rules in this module must generalise across cards. Do not hard-code a specific
person, address, inspector, passport number, or date. Allowed logic:
  * type validation: dates, passport numbers, serials, registration numbers;
  * cleanup of labels/noise;
  * lexicon matching for countries/cities;
  * canonicalisation of fixed printed abbreviations such as ВКД / ХШБ ВКД ҶТ.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import Config
from .tajik_normalize import fuzzy_match_lexicon, get_field_values, normalize_text

DATE_RE = re.compile(r'(?<!\d)(\d{1,2})[./,\-\s]+(\d{1,2})[./,\-\s]+(\d{2,4})(?!\d)')
DIGIT_FIX = str.maketrans({
    'O': '0', 'О': '0', 'o': '0', 'о': '0',
    'I': '1', 'І': '1', 'l': '1', 'L': '1', '|': '1',
    'S': '5', 'З': '3', 'з': '3', 'B': '8',
})
CYR_TO_LATIN_PASSPORT = str.maketrans({
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H',
    'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X',
})

LABEL_WORD_RE = re.compile(
    r'\b(ШАҲРВАНД[ӢИ]|ШАХРВАНД[ИӢ]|НОМУ?|НАСАБ[И]?|РАҚАМИ|РИҚАМИ|ШИНОСНОМА[И]?|'
    r'САНАИ|БАҚАЙДГИР[ӢИ]|БАКАЙДГИР[ИӢ]|ҶОИ|ЧОИ|ЗИСТ|МӮҲЛАТ|МУХЛАТ|ИСТИҚОМАТ|'
    r'ЭЪТИБОР|СИЛСИЛА|НАЗОРАТ|НОЗИР|ИНСПЕКТОР|ТАМДИД|ВАЗОРАТИ|КОРҲОИ|ДОХИЛӢ|ИДОМА)\b',
    re.I,
)
SETTLEMENT_RE = re.compile(
    r'(ш\.?|шаҳр|шахр|ноҳия|нохия|ҷамоат|чамоат|деҳа|деха|кӯч|куч|вил\.?|вилоят|н\.)',
    re.I,
)
NOISE_WORDS_RE = re.compile(r'\b(шиноснома[и]?|шаҳрванд[ӣи]?|шахрванд[и]?|бақайдгир[ӣи]|регистрац)\b', re.I)


def _field_template(key: str, value: str, confidence: float = 0.78, source: str = 'postprocess') -> Dict[str, Any]:
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


def _digits(value: str) -> str:
    return re.sub(r'\D', '', value or '')


def _similar_digits(a: str, b: str) -> float:
    a_digits, b_digits = _digits(a), _digits(b)
    if not a_digits or not b_digits:
        return 0.0
    return SequenceMatcher(None, a_digits, b_digits).ratio()


def _raw_text(ocr_results: Iterable[Dict[str, Any]]) -> str:
    return ' '.join(str(r.get('text', '')) for r in (ocr_results or []))


def _bbox_center(block: Dict[str, Any]) -> Tuple[float, float]:
    bbox = block.get('bbox') or [[0, 0], [0, 0]]
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _looks_like_noise(value: str) -> bool:
    value = value or ''
    words = [w for w in re.split(r'\s+', value.strip()) if w]
    if not words:
        return True
    one_char = sum(1 for w in words if len(w.strip('.')) <= 1)
    return len(words) >= 5 and one_char / len(words) > 0.35


def _norm_date(text: str) -> Optional[str]:
    match = DATE_RE.search((text or '').translate(DIGIT_FIX))
    if not match:
        return None
    day, month, year = match.groups()
    day, month = day.zfill(2), month.zfill(2)
    if len(year) == 2:
        year = '20' + year if int(year) < 80 else '19' + year
    if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
        return f'{day}.{month}.{year}'
    return None


def _looks_like_date_or_year(value: str, dates: Iterable[str]) -> bool:
    digits = _digits(value)
    if not digits:
        return False
    if any(year in digits for year in ('2023', '2024', '2025', '2026', '2027', '2028')):
        return True
    return any(digits and (digits in _digits(d) or _digits(d) in digits) for d in dates if d)


def _normalise_passport(raw_candidate: str) -> Optional[str]:
    text = str(raw_candidate or '').strip()

    # Booklet passports: 1-3 latin/cyrillic letters + 6-9 digits.
    compact = re.sub(r'[^A-Za-zА-Яа-я0-9]', '', text).upper()
    match = re.fullmatch(r'([A-ZА-Я]{1,3})[CcСс]?(\d{6,9})', compact)
    if match:
        prefix = match.group(1).translate(CYR_TO_LATIN_PASSPORT)
        # N/H/Н is often OCR for the № sign, not a true booklet prefix.
        if prefix not in {'N', 'H'}:
            return prefix + match.group(2)

    # Internal passport line: № or OCR variants N/H/Н + digits.
    norm = text.translate(DIGIT_FIX).upper().replace('№', 'N').replace('Н', 'N').replace('H', 'N')
    match = re.search(r'N\s*([0-9]{6,9})', norm)
    if match:
        digits = match.group(1)
        if len(digits) > 8:
            digits = digits[-8:]
        if 6 <= len(digits) <= 8:
            return '№' + digits
    return None


def _find_passport(raw: str, current: str) -> str:
    candidates: List[Tuple[float, str]] = []

    for match in re.finditer(r'\b([A-ZА-Я]{1,3})[CcСс]?\s?\d{6,9}\b', raw, flags=re.I):
        value = _normalise_passport(match.group(0))
        if value and not value.startswith('№'):
            score = 1.0 + min(len(_digits(value)), 9) / 20
            candidates.append((score, value))

    for match in re.finditer(r'(?:№|N|Н|H)\s*[0-9OОoоIІlL|BЗзS]{6,9}', raw, flags=re.I):
        value = _normalise_passport(match.group(0))
        if value:
            score = 0.9 + min(len(_digits(value)), 8) / 20
            candidates.append((score, value))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return current or ''


def _is_passport_line(raw_text: str, roi_field: Optional[str]) -> bool:
    """Return True if a numeric block belongs to passport, not card number.

    The top card number can be printed after № too, so never reject № inside the
    explicit registration_card_number ROI. Outside that ROI, №/N/H + digits is
    most likely the passport line.
    """
    if roi_field == 'registration_card_number':
        return False
    return bool(re.search(r'(№|\bN\s*\d|\bН\s*\d|\bH\s*\d)', raw_text, flags=re.I))


def _find_best_registration_number(ocr_results: Iterable[Dict[str, Any]], current: str, passport: str, dates: Iterable[str]) -> str:
    candidates: List[Tuple[float, str]] = []
    passport_digits = _digits(passport)

    for block in ocr_results or []:
        raw = str(block.get('text', ''))
        roi_field = block.get('roi_field')
        if _is_passport_line(raw, roi_field):
            continue

        text = raw.translate(DIGIT_FIX)
        for match in re.finditer(r'(?<!\d)(\d{6,8})(?!\d)', text):
            value = match.group(1)
            if _looks_like_date_or_year(value, dates):
                continue
            if passport_digits and (
                value in passport_digits or passport_digits in value or _similar_digits(value, passport_digits) >= 0.72
            ):
                continue

            _, y = _bbox_center(block)
            conf = float(block.get('confidence', 0.5))
            score = conf
            if len(value) == 7:
                score += 1.2
            elif len(value) == 8:
                score -= 0.4
            if y < 1200:
                score += 0.2
            if roi_field == 'registration_card_number':
                score += 1.5
            candidates.append((score, value))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    if current and not _looks_like_date_or_year(current, dates):
        return current
    return ''


def _find_best_serial(ocr_results: Iterable[Dict[str, Any]], current: str, passport: str, reg: str, dates: Iterable[str]) -> str:
    candidates: List[Tuple[float, str]] = []
    for block in ocr_results or []:
        text = str(block.get('text', '')).translate(DIGIT_FIX)
        for match in re.finditer(r'(?<!\d)(\d{3,5})(?!\d)', text):
            value = match.group(1)
            if value in {'2023', '2024', '2025', '2026', '2027', '2028'}:
                continue
            if passport and value in _digits(passport):
                continue
            if reg and value in _digits(reg):
                continue
            if any(value in _digits(d) for d in dates if d):
                continue

            x, y = _bbox_center(block)
            conf = float(block.get('confidence', 0.5))
            score = conf
            if len(value) == 3:
                score += 0.7
            if 0.42 <= (y / 3600.0) <= 0.62 and x > 1000:
                score += 0.6
            if block.get('roi_field') == 'serial_control_number':
                score += 0.4
            candidates.append((score, value))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    if current and current not in {'2024', '2025', '2026', '2027'}:
        return current
    return ''


def _clean_person_name(value: str) -> str:
    value = LABEL_WORD_RE.sub(' ', value or '')
    value = DATE_RE.sub(' ', value)
    value = re.sub(r'\b[A-ZА-ЯЁ]{1,3}\d{4,9}\b', ' ', value, flags=re.I)
    value = re.sub(r'\d+', ' ', value)
    value = re.sub(r'[^\wА-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ\s\.\-]', ' ', value)
    parts = [p for p in re.split(r'\s+', value.strip()) if p]
    while parts and (len(parts[-1].strip('.')) < 2 or not re.search(r'[А-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ]', parts[-1])):
        parts.pop()
    return ' '.join(parts[:4]).strip(' .-')


def _clean_inspector(value: str) -> str:
    value = LABEL_WORD_RE.sub(' ', value or '')
    value = DATE_RE.sub(' ', value)
    value = re.sub(r'\d+', ' ', value)
    value = re.sub(r'[^А-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ\s\.]', ' ', value)
    tokens = [p.strip('.') for p in re.split(r'\s+', value.strip()) if p.strip('.')]
    surnames = [t for t in tokens if len(t) >= 3]
    if not surnames:
        return ''
    start = tokens.index(surnames[0])
    kept = [surnames[0]]
    for token in tokens[start + 1:]:
        kept.append(token)
        if len(kept) >= 3:
            break
    return ' '.join(kept).strip(' .-')


def _match_city(text: str) -> Optional[str]:
    cities = get_field_values('cities')
    if not cities or not text:
        return None
    for word in re.findall(r'[А-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ]{3,}', text):
        match = fuzzy_match_lexicon(word, cities, threshold=0.82)
        if match:
            return match
    return None


def _normalize_citizenship_value(value: str) -> str:
    value = normalize_text((value or '').strip(' .:;,-()'))
    if not value:
        return ''
    match = fuzzy_match_lexicon(value, get_field_values('countries'), threshold=0.8)
    return match if match else value


def postprocess_fields(parsed_data: Dict[str, Dict[str, Any]], ocr_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Apply generic sanity checks to the parsed field map."""
    raw = _raw_text(ocr_results)
    raw_l = raw.lower()

    passport = _find_passport(raw, parsed_data.get('passport_number', {}).get('value', ''))
    if passport:
        parsed_data['passport_number'] = _field_template('passport_number', passport, 0.86)

    citizenship = parsed_data.get('citizenship', {}).get('value', '')
    if citizenship:
        normalized = _normalize_citizenship_value(citizenship)
        parsed_data['citizenship'] = _field_template(
            'citizenship', normalized, parsed_data.get('citizenship', {}).get('confidence', 0.8)
        )

    name = parsed_data.get('name_and_surname', {}).get('value', '')
    if name:
        cleaned = _clean_person_name(name)
        if cleaned and not _looks_like_noise(cleaned):
            parsed_data['name_and_surname'] = _field_template('name_and_surname', cleaned, 0.82)
        else:
            parsed_data['name_and_surname'] = _field_template('name_and_surname', '', 0.0)

    prs = parsed_data.get('prs_mia_rt', {}).get('value', '')
    if ('хшб' in raw_l or 'вкд' in raw_l) and (not prs or 'вкд' not in prs.lower() or 'хшб' not in prs.lower()):
        parsed_data['prs_mia_rt'] = _field_template('prs_mia_rt', 'ХШБ ВКД ҶТ', 0.82)
    elif prs and not re.search(r'(хшб|вкд|ҷт|чт)', prs, flags=re.I):
        parsed_data['prs_mia_rt'] = _field_template('prs_mia_rt', '', 0.0)

    mia = parsed_data.get('mia', {}).get('value', '')
    if re.search(r'\bвкд\b', raw_l):
        parsed_data['mia'] = _field_template('mia', 'ВКД', 0.85)
    elif mia and (len(mia) > 12 or _looks_like_noise(mia)):
        parsed_data['mia'] = _field_template('mia', '', 0.0)

    all_dates: List[Tuple[float, str, float]] = []
    for block in ocr_results or []:
        date_value = _norm_date(str(block.get('text', '')))
        if date_value:
            _, y = _bbox_center(block)
            all_dates.append((y, date_value, float(block.get('confidence', 0.6))))
    all_dates.sort(key=lambda row: row[0])
    unique_dates: List[Tuple[str, float]] = []
    for _, date_value, conf in all_dates:
        if date_value not in [d for d, _ in unique_dates]:
            unique_dates.append((date_value, conf))

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
    registration_number = _find_best_registration_number(
        ocr_results, parsed_data.get('registration_card_number', {}).get('value', ''), passport, dates
    )
    parsed_data['registration_card_number'] = _field_template(
        'registration_card_number', registration_number, 0.86 if registration_number else 0.0
    )

    serial = _find_best_serial(
        ocr_results, parsed_data.get('serial_control_number', {}).get('value', ''), passport, registration_number, dates
    )
    parsed_data['serial_control_number'] = _field_template('serial_control_number', serial, 0.84 if serial else 0.0)

    place = parsed_data.get('place_of_residence', {}).get('value', '')
    city = _match_city(raw)
    if city:
        parsed_data['place_of_residence'] = _field_template('place_of_residence', f'ш. {city}', 0.84)
    elif place and (NOISE_WORDS_RE.search(place) or _looks_like_noise(place) or not SETTLEMENT_RE.search(place)):
        parsed_data['place_of_residence'] = _field_template('place_of_residence', '', 0.0)

    residence_cont = parsed_data.get('place_of_residence_cont', {}).get('value', '')
    if residence_cont and (len(residence_cont) > 40 or DATE_RE.search(residence_cont) or _looks_like_noise(residence_cont)):
        parsed_data['place_of_residence_cont'] = _field_template('place_of_residence_cont', '', 0.0)

    inspector = parsed_data.get('inspector', {}).get('value', '')
    if inspector:
        cleaned = _clean_inspector(inspector)
        if cleaned and not _looks_like_noise(cleaned):
            parsed_data['inspector'] = _field_template('inspector', cleaned, 0.8)
        else:
            parsed_data['inspector'] = _field_template('inspector', '', 0.0)

    return parsed_data
