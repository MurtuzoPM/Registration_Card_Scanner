"""Generic, card-agnostic post-processing for registration-card fields.

This is the single source of truth used by the Flask app AND the offline
tools (debug_extract.py, evaluate_dataset.py).  Keeping it in one place avoids
the drift that previously made the evaluation score disagree with the app.

Design rule: NO card-specific answers.  Nothing here may hard-code the value of
a particular person's name, address, inspector or passport.  Only these kinds
of corrections are allowed because they generalise to every card of this type:

  * Spelling/encoding normalisation of Tajik letters and common OCR confusions.
  * Validation by field *type* (a date looks like a date, a passport like a
    passport, ...).
  * Matching free text against a *lexicon* of known countries / cities.
  * Canonicalising the fixed text that is *printed on every RT card*
    (e.g. "ХШБ ВКД ҶТ" and "ВКД"), when OCR returns a recognisable fragment.

Everything else (names, addresses, inspectors, ID numbers, dates) must come
from the OCR of the specific card being scanned.
"""

import re
from difflib import SequenceMatcher

from config import Config
from .tajik_normalize import normalize_text, get_field_values, fuzzy_match_lexicon

# ---------------------------------------------------------------------------
# Shared regex / translation tables
# ---------------------------------------------------------------------------
DATE_RE = re.compile(r'(?<!\d)(\d{1,2})[./,\-\s]+(\d{1,2})[./,\-\s]+(\d{2,4})(?!\d)')

DIGIT_FIX = str.maketrans({
    'O': '0', 'О': '0', 'o': '0', 'о': '0',
    'I': '1', 'І': '1', 'l': '1', 'L': '1', '|': '1',
    'S': '5', 'З': '3', 'з': '3', 'B': '8',
})

# Cyrillic label words that must never be kept as part of a person/place value.
LABEL_WORD_RE = re.compile(
    r'\b(ШАҲРВАНД[ӢИ]|ШАХРВАНД[ИӢ]|НОМУ?|НАСАБ[И]?|РАҚАМИ|РИҚАМИ|ШИНОСНОМА[И]?|'
    r'САНАИ|БАҚАЙДГИР[ӢИ]|БАКАЙДГИР[ИӢ]|ҶОИ|ЧОИ|ЗИСТ|МӮҲЛАТ|МУХЛАТ|ИСТИҚОМАТ|'
    r'ЭЪТИБОР|СИЛСИЛА|НАЗОРАТ|НОЗИР|ИНСПЕКТОР|ТАМДИД|ВАЗОРАТИ|КОРҲОИ|ДОХИЛӢ|ИДОМА)\b',
    re.I,
)

# Markers that indicate a string is a real settlement / address line.
SETTLEMENT_RE = re.compile(
    r'(ш\.?|шаҳр|шахр|ноҳия|нохия|ҷамоат|чамоат|деҳа|деха|кӯч|куч|вил\.?|вилоят|н\.)',
    re.I,
)


def _digits(value):
    return re.sub(r'\D', '', value or '')


def _similar_digits(a, b):
    a, b = _digits(a), _digits(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _bbox_center(block):
    bbox = block.get('bbox') or [[0, 0], [0, 0]]
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _raw_text(ocr_results):
    return ' '.join(str(r.get('text', '')) for r in (ocr_results or []))


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
    return {'value': value, 'confidence': confidence, 'label': key,
            'field_number': None, 'source': source}


def _looks_like_noise(value):
    value = value or ''
    words = [w for w in re.split(r'\s+', value.strip()) if w]
    if not words:
        return True
    one_char = sum(1 for w in words if len(w) <= 1)
    return len(words) >= 5 and one_char / len(words) > 0.35


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
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
    if any(year in digits for year in ('2023', '2024', '2025', '2026', '2027', '2028')):
        return True
    return any(digits and (digits in _digits(d) or _digits(d) in digits)
               for d in dates if d)


# ---------------------------------------------------------------------------
# Passport (generic — supports letter-prefixed and №/digit internal passports)
# ---------------------------------------------------------------------------
def _normalise_passport(raw_candidate):
    """Return a normalised passport string or None.

    Accepts two real-world forms without hard-coding any specific number:
      * Letter-prefixed booklet passports, e.g. ``EC5671065`` / ``P06476028``
      * Internal "№" passports, e.g. ``№0011801`` (OCR often drops the №).

    A lone ``N`` / ``Н`` / ``H`` before the digits is treated as the "№" sign,
    not a booklet prefix (those are never valid Tajik booklet prefixes).
    """
    text = str(raw_candidate or '').strip()

    # Form 1: 1-2 latin/cyrillic letters + 6-9 digits (booklet passport).
    letter_form = re.sub(r'[^A-Za-zА-Яа-я0-9]', '', text)
    m = re.match(r'^([A-ZА-Я]{1,2})[CcСс]?\s*(\d{6,9})$', letter_form.upper())
    if m and m.group(1) not in {'N', 'Н', 'H'}:
        prefix = m.group(1).translate(str.maketrans('АВЕКМНОРСТХ', 'ABEKMHOPCTX'))
        return prefix + m.group(2)

    # Form 2: №/N + digits (internal passport). The leading "№" is written as
    # "N" to match the convention used on these cards (e.g. N00085561).
    norm = text.translate(DIGIT_FIX).upper().replace('№', 'N').replace('Н', 'N').replace('H', 'N')
    m = re.search(r'N\s*([0-9]{6,9})', norm)
    if m:
        digits = m.group(1)
        if len(digits) > 8:
            digits = digits[-8:]
        if 6 <= len(digits) <= 8:
            return 'N' + digits
    return None


def _find_passport(raw, current):
    # 1) Booklet passports: a real 1-2 letter prefix (not the №-sign) + digits.
    for m in re.finditer(r'\b([A-ZА-Я]{1,2})[CcСс]?\s?\d{6,9}\b', raw, flags=re.I):
        if m.group(1).upper() in {'N', 'Н', 'H'}:
            continue
        value = _normalise_passport(m.group(0))
        if value and not value.startswith('№'):
            return value
    # 2) Internal "№" passports (the №/N may be OCR noise).
    for m in re.finditer(r'(?:№|N|Н|H)\s*[0-9OОoоIІlL|BЗзS]{6,9}', raw, flags=re.I):
        value = _normalise_passport(m.group(0))
        if value:
            return value
    return current


# ---------------------------------------------------------------------------
# Registration card number / serial number (generic numeric selection)
# ---------------------------------------------------------------------------
def _find_best_registration_number(ocr_results, current, passport, dates):
    candidates = []
    passport_digits = _digits(passport)
    for block in ocr_results or []:
        raw_text = str(block.get('text', ''))
        text = raw_text.translate(DIGIT_FIX)
        # The passport line (№/N...) must not become the card number.
        if re.search(r'(№|\bN\s*\d|\bН\s*\d)', raw_text, flags=re.I):
            continue
        for m in re.finditer(r'(?<!\d)(\d{6,8})(?!\d)', text):
            val = m.group(1)
            if _looks_like_date_or_year(val, dates):
                continue
            if passport_digits and (
                val in passport_digits or passport_digits in val
                or _similar_digits(val, passport_digits) >= 0.72
            ):
                continue
            _, y = _bbox_center(block)
            conf = float(block.get('confidence', 0.5))
            score = conf
            if len(val) == 7:
                score += 1.2
            elif len(val) == 8:
                score -= 0.4
            if y < 1200:
                score += 0.2
            if block.get('roi_field') == 'registration_card_number':
                score += 1.5
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
            if val in {'2023', '2024', '2025', '2026', '2027', '2028'}:
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
                score += 0.7
            if 0.42 <= (y / 3600.0) <= 0.62 and x > 1000:
                score += 0.6
            if block.get('roi_field') == 'serial_control_number':
                score += 0.4
            candidates.append((score, val))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    if current and current not in {'2024', '2025', '2026', '2027'}:
        return current
    return ''


# ---------------------------------------------------------------------------
# Generic text cleanup for name / inspector / place
# ---------------------------------------------------------------------------
def _clean_person_name(value):
    """Strip label words / digits / codes from a name without hard-coding it."""
    value = LABEL_WORD_RE.sub(' ', value or '')
    value = DATE_RE.sub(' ', value)
    value = re.sub(r'\b[A-ZА-ЯЁ]{1,3}\d{4,9}\b', ' ', value, flags=re.I)
    value = re.sub(r'\d+', ' ', value)
    value = re.sub(r'[^\wА-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ\s\.\-]', ' ', value)
    parts = [p for p in re.split(r'\s+', value.strip()) if p]
    # Drop trailing single letters / non-cyrillic fragments.
    while parts and (len(parts[-1].strip('.')) < 2
                     or not re.search(r'[А-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ]', parts[-1])):
        parts.pop()
    return ' '.join(parts[:4]).strip(' .-')


def _clean_inspector(value):
    """Keep a Cyrillic surname (and optional initial) without hard-coding it.

    Requires at least one surname-like token (>= 3 letters); a value made up of
    only single-letter fragments is treated as noise and dropped.
    """
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
    for t in tokens[start + 1:]:
        kept.append(t)
        if len(kept) >= 3:
            break
    return ' '.join(kept).strip(' .-')


def _match_city(text):
    """Return a known city found anywhere in *text*, else None (no forcing)."""
    cities = get_field_values('cities')
    if not cities or not text:
        return None
    for word in re.findall(r'[А-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ]{3,}', text):
        match = fuzzy_match_lexicon(word, cities, threshold=0.82)
        if match:
            return match
    return None


def _normalize_citizenship_value(value):
    """Normalise spelling and map to a known country via the lexicon only."""
    value = normalize_text((value or '').strip(' .:;,-()'))
    if not value:
        return ''
    match = fuzzy_match_lexicon(value, get_field_values('countries'), threshold=0.8)
    return match if match else value


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def postprocess_fields(parsed_data, ocr_results):
    """Apply generic, card-agnostic sanity checks to the parsed field map."""
    raw = _raw_text(ocr_results)
    raw_l = raw.lower()

    # --- Passport -------------------------------------------------------
    passport = _find_passport(raw, parsed_data.get('passport_number', {}).get('value', ''))
    if passport:
        parsed_data['passport_number'] = _field_template('passport_number', passport, 0.86)

    # --- Citizenship (lexicon only; no forced country) ------------------
    cit = parsed_data.get('citizenship', {}).get('value', '')
    if cit:
        norm_cit = _normalize_citizenship_value(cit)
        if norm_cit:
            parsed_data['citizenship'] = _field_template(
                'citizenship', norm_cit,
                parsed_data.get('citizenship', {}).get('confidence', 0.8),
            )

    # --- Name (generic cleanup) -----------------------------------------
    name = parsed_data.get('name_and_surname', {}).get('value', '')
    if name:
        cleaned = _clean_person_name(name)
        if not cleaned or _looks_like_noise(cleaned):
            parsed_data['name_and_surname'] = _field_template('name_and_surname', '', 0.0)
        else:
            parsed_data['name_and_surname'] = _field_template('name_and_surname', cleaned, 0.82)

    # --- PRS MIA RT / MIA: canonicalise the constant printed text -------
    # These abbreviations are printed identically on every RT registration
    # card, so recognising a fragment and canonicalising is card-agnostic.
    prs = parsed_data.get('prs_mia_rt', {}).get('value', '')
    if ('хшб' in raw_l or 'вкд' in raw_l) and (
        not prs or 'вкд' not in prs.lower() or 'хшб' not in prs.lower()
    ):
        parsed_data['prs_mia_rt'] = _field_template('prs_mia_rt', 'ХШБ ВКД ҶТ', 0.82)
    elif prs and not re.search(r'(хшб|вкд|ҷт|чт)', prs, flags=re.I):
        parsed_data['prs_mia_rt'] = _field_template('prs_mia_rt', '', 0.0)

    mia = parsed_data.get('mia', {}).get('value', '')
    if re.search(r'\bвкд\b', raw_l):
        parsed_data['mia'] = _field_template('mia', 'ВКД', 0.85)
    elif mia and (len(mia) > 12 or _looks_like_noise(mia)):
        parsed_data['mia'] = _field_template('mia', '', 0.0)

    # --- Dates (position-based assignment, fully generic) ---------------
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
    if (reg_date and valid_until and reg_date == valid_until
            and parsed_data.get('date_of_registration', {}).get('source')
            not in {'roi', 'position_date', 'postprocess'}):
        parsed_data['date_of_registration'] = _field_template('date_of_registration', '', 0.0)
    if valid_until:
        earlier = [d for d, _ in unique_dates if d != valid_until]
        if earlier and (not parsed_data.get('date_of_registration', {}).get('value')
                        or parsed_data['date_of_registration']['value'] == valid_until):
            parsed_data['date_of_registration'] = _field_template(
                'date_of_registration', earlier[0], 0.84)
    if (not parsed_data.get('date_of_registration_extension', {}).get('value')
            and parsed_data.get('date_of_registration', {}).get('value')):
        parsed_data['date_of_registration_extension'] = _field_template(
            'date_of_registration_extension',
            parsed_data['date_of_registration']['value'], 0.72)

    # --- Registration card number & serial (generic numeric selection) --
    dates = [
        parsed_data.get('date_of_registration', {}).get('value', ''),
        parsed_data.get('valid_until', {}).get('value', ''),
    ]
    passport = parsed_data.get('passport_number', {}).get('value', '')
    reg = _find_best_registration_number(
        ocr_results,
        parsed_data.get('registration_card_number', {}).get('value', ''),
        passport, dates,
    )
    parsed_data['registration_card_number'] = _field_template(
        'registration_card_number', reg, 0.86 if reg else 0.0)

    serial = _find_best_serial(
        ocr_results,
        parsed_data.get('serial_control_number', {}).get('value', ''),
        passport, reg, dates,
    )
    parsed_data['serial_control_number'] = _field_template(
        'serial_control_number', serial, 0.84 if serial else 0.0)

    # --- Place of residence (city lexicon; generic continuation) --------
    place = parsed_data.get('place_of_residence', {}).get('value', '')
    city = _match_city(raw)
    if city:
        parsed_data['place_of_residence'] = _field_template(
            'place_of_residence', f'ш. {city}', 0.84)
    elif place and not SETTLEMENT_RE.search(place):
        # Existing value is not a recognisable settlement -> drop it.
        if re.search(r'шиноснома|шаҳрванд|паспорт', place, re.I) or _looks_like_noise(place):
            parsed_data['place_of_residence'] = _field_template('place_of_residence', '', 0.0)

    cont = parsed_data.get('place_of_residence_cont', {}).get('value', '')
    if cont and (len(cont) > 40 or DATE_RE.search(cont) or _looks_like_noise(cont)):
        parsed_data['place_of_residence_cont'] = _field_template('place_of_residence_cont', '', 0.0)

    # --- Inspector (generic surname extraction) -------------------------
    inspector = parsed_data.get('inspector', {}).get('value', '')
    if inspector:
        cleaned = _clean_inspector(inspector)
        if cleaned and not _looks_like_noise(cleaned):
            parsed_data['inspector'] = _field_template('inspector', cleaned, 0.8)
        else:
            parsed_data['inspector'] = _field_template('inspector', '', 0.0)

    return parsed_data
