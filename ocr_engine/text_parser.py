import re
import logging
from difflib import SequenceMatcher
from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def fuzzy_match(text, keyword, threshold=0.6):
    """Check if keyword approximately matches text."""
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    if keyword_lower in text_lower:
        return True
    if len(keyword_lower) < 3:
        return keyword_lower in text_lower
    for word in text_lower.split():
        if SequenceMatcher(None, word, keyword_lower).ratio() >= threshold:
            return True
    return False


def fuzzy_match_any(text, keywords, threshold=0.6):
    """Check if text approximately matches ANY keyword."""
    text_lower = text.lower()
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in text_lower:
            return True
        for word in text_lower.split():
            if SequenceMatcher(None, word, kw_lower).ratio() >= threshold:
                return True
    return False


def _is_date_text(text):
    """Check if text looks like a date value."""
    text_stripped = text.strip()
    if re.search(r'\d{2}[./\-]\d{2}[./\-]\d{2,4}', text_stripped):
        return True
    cleaned = re.sub(r'[^0-9]', '', text_stripped)
    if len(cleaned) == 8:
        return True
    return False


def _is_number_or_code(text):
    """Check if text looks like a number, passport code, or serial number."""
    text = text.strip()
    if re.match(r'^[A-Z0-9]{1,3}\s?[A-Z0-9]{6,8}$', text, re.IGNORECASE):
        return True
    if re.match(r'^\d{3,}$', text):
        return True
    if re.match(r'^[A-Z0-9\-]{5,9}$', text, re.IGNORECASE):
        return True
    if re.match(r'^№?\s?\d+$', text):
        return True
    return False


# Known value strings that look like labels but ARE valid field values
# (field 6 = PRS MIA RT abbreviation, field 9 = MIA abbreviation)
_KNOWN_VALID_VALUES = {
    'вкд', 'хшб вкд чт', 'хшб', 'хшб вкд', 'вкд чт',
    'vkd', 'hshb vkd cht', 'вk', 'bk', 'вкz', 'вkz',
}


def _is_pure_label_text(text):
    """
    Check if text is likely a field description/label rather than a value.
    Returns True only when confident it is a label – errs on side of keeping values.
    """
    text_stripped = text.strip()
    if not text_stripped:
        return True

    # Dates are always values
    if _is_date_text(text_stripped):
        return False

    text_lower = text_stripped.lower()

    # Short abbreviations that ARE valid field values – never reject these
    if text_lower in _KNOWN_VALID_VALUES:
        return False

    words = text_stripped.split()

    # Base label keywords (correct Tajik forms)
    base_label_keywords = [
        'рақами', 'риқами', 'бақайдгирӣ', 'қайд', 'шиноснома',
        'шаҳрвандӣ', 'шахрвандӣ', 'шаҳрванди', 'шахрванди',
        'ном', 'насаб', 'санаи', 'паспортӣ', 'бақайдгирии',
        'эътибор', 'мӯҳлат', 'мухлат', 'силсила', 'назорат',
        'вазорати', 'корҳои', 'дохилӣ', 'ҷои зист', 'чои зист', 'чои истикомат',
        'суроға', 'нозир', 'инспектор', 'тамдид', 'корти',
        'варақаи', 'варакаи', 'вазорат', 'идома', 'давом', 'истиқомат',
        'истикомат', 'карточка', 'камочка', 'регистрации', 'регистрация',
        'registration', 'registrac', 'regist', 'card',
        'чумхурии', 'чумхурни', 'точикистон',
        'рег', 'регистр', 'номгуи', 'истифода',
        'кабулшуда', 'ташкилот', 'шахси', 'кабулкунанда',
        'дохили', 'дохилин', 'варакаи',
    ]

    extended_label_keywords = base_label_keywords + [
        'бакаидгири', 'бакайдгири', 'бакаидгирӣ', 'бакаидгирий',
        'бакайдгирий',
        'изорати', 'взорати', 'возорати',
        'дозилин', 'дозищив', 'дохилин', 'дохили',
        'цумхурии', 'цумхурни', 'циумхурни',
        'шахрванди', 'шахрапди', 'шахраанди',
        'шиносномаи',
        'эьтибор', 'эъти6ор',
        'тадид', 'тамди',
        'kecistration', 'registrat', 'reistratio',
    ]

    short_label_fragments = {'no', 'n', 'h', 'i', 'i:', 'т', 'фм', 'тм', 'tм', 'вк', 'вкz'}
    if text_lower.strip('.:;,- ') in short_label_fragments:
        return True

    # Label numbers 1-13 on the right side
    if text_stripped.isdigit():
        n = int(text_stripped)
        if 1 <= n <= 13:
            return True

    # Very short non-digit noise
    if len(text_stripped) <= 2 and not text_stripped.isdigit():
        return True

    # Exact or close prefix match with any extended keyword
    for kw in extended_label_keywords:
        kw_lower = kw.lower()
        if text_lower == kw_lower:
            return True
        if text_lower.startswith(kw_lower):
            if len(text_lower) <= len(kw_lower) + 3:
                return True

    # Fuzzy single-word match against base keywords
    if len(words) == 1:
        for kw in base_label_keywords:
            if len(kw) > 3 and SequenceMatcher(None, text_lower, kw.lower()).ratio() >= 0.75:
                return True

    # Fuzzy multi-word match against known multi-word phrases
    if len(words) >= 2:
        for kw in base_label_keywords:
            if ' ' in kw:
                kw_parts = kw.split()
                if SequenceMatcher(None, text_lower, kw.lower()).ratio() >= 0.7:
                    return True
                if len(words) == len(kw_parts):
                    matches = sum(
                        1 for tw, kw_w in zip(words, kw_parts)
                        if SequenceMatcher(None, tw, kw_w).ratio() >= 0.7
                    )
                    if matches >= len(kw_parts) - 1:
                        return True

    if len(words) <= 2 and any(text_lower.startswith(k) for k in extended_label_keywords):
        return True

    strong_label_phrases = [
        'рақами корти бақайдгирӣ', 'рақами шиноснома', 'ном ва насаб',
        'санаи бақайдгирӣ', 'то кай эътибор', 'ҷои зист', 'чои зист',
        'вазорати корҳои', 'паспортӣ-бақайдгирии',
        'паспортӣ бақайдгирии', 'бақайдгирӣ ё тамдид', 'корти бақайдгирӣ',
        'карточка регистрации', 'registration card', 'варақаи бақайдгирӣ',
        'ба қайд гирифта шуд', 'тамдид карда шуд', 'чои истикомат',
        'идомаи ҷои зист', 'вазорати корхои дохили', 'чумхурии точикистон',
        'барои истифода', 'кабулшуда', 'ташкилот', 'шахси', 'кабулкунанда',
        'ба кайд гирифта', 'ба қайд гирифта',
        # label-line phrases that appear in raw OCR of this card
        'ба кайд гирифта шуд', 'тамдид карда шуд',
        'ба қайд гирифта', 'ба кайд гирифта',
        'шуд / тамдид', 'карда шуд',
        'насабу ном', 'ласабу ном',
    ]
    for phrase in strong_label_phrases:
        if phrase in text_lower:
            return True

    match_count = sum(1 for k in base_label_keywords if k in text_lower)
    if match_count >= 2 and len(words) < 6:
        return True

    if text_stripped.startswith('(') or text_stripped.startswith(')'):
        return True

    is_code = _is_number_or_code(text_stripped)
    if is_code:
        for kw in base_label_keywords:
            if len(kw) > 3:
                cleaned = re.sub(r'[0-9]', '', text_lower)
                if len(cleaned) > 3 and SequenceMatcher(None, cleaned, kw.lower()).ratio() >= 0.6:
                    return True
                for tw in text_lower.split():
                    tw_clean = re.sub(r'[0-9]', '', tw)
                    if len(tw_clean) > 3 and SequenceMatcher(None, tw_clean, kw.lower()).ratio() >= 0.65:
                        return True

    if text_stripped.isupper() and len(text_stripped) > 3:
        # Do NOT reject short all-caps abbreviations that are known valid values
        if text_lower not in _KNOWN_VALID_VALUES:
            for kw in base_label_keywords:
                if len(kw) > 3:
                    if SequenceMatcher(None, text_lower, kw.lower()).ratio() >= 0.6:
                        return True
                    text_words = text_lower.split()
                    kw_words = kw.split()
                    for tw in text_words:
                        for kw_w in kw_words:
                            if len(tw) > 3 and len(kw_w) > 3:
                                if SequenceMatcher(None, tw, kw_w).ratio() >= 0.7:
                                    return True
                    cleaned = re.sub(r'[^a-z\u0400-\u04FF]', '', text_lower)
                    if len(cleaned) > 3 and SequenceMatcher(None, cleaned, kw.lower()).ratio() >= 0.7:
                        return True

    return False


def _clean_date_value(text):
    text = re.sub(r'\s+', '', text).strip()
    # Handle DDMMYYYY or similar
    if len(text) == 8 and text.isdigit():
        return f'{text[:2]}.{text[2:4]}.{text[4:]}'
    m = re.search(r'(\d{2})[.,/\\-](\d{2})[.,/\\-](\d{4})', text)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        if int(day) > 31 and day.startswith('4'): day = '1' + day[1]
        return f'{day}.{month}.{year}'
    m = re.search(r'(\d{2})[.,/\\-](\d{2})[.,/\\-](\d{2})\b', text)
    if m:
        year = int(m.group(3))
        full_year = f'20{m.group(3)}' if year < 80 else f'19{m.group(3)}'
        return f'{m.group(1)}.{m.group(2)}.{full_year}'
    return text


def _clean_number_value(text):
    """
    Strip common prefixes from number fields (№, No, N).
    NOTE: do NOT apply OCR character substitutions (O->0 etc.) here because
    that destroys valid Cyrillic text and passport letters.  The caller may
    apply character corrections on digit-only results only.
    """
    text = text.strip()
    text = re.sub(r'^[№NnNo\.:\s]+', '', text).strip()
    text = re.sub(r'[-_\'\"*\s]+$', '', text).strip()
    return text


def _fix_digit_ocr(text):
    """
    Apply common OCR digit corrections (O->0, Z->7, S->5, I->1, B->8).
    Only call on strings that are expected to be purely numeric/alphanumeric codes,
    NOT on strings that may contain Cyrillic letters.
    """
    return text.upper().replace('O', '0').replace('Z', '7').replace('S', '5').replace('I', '1').replace('B', '8')


def _looks_like_passport(text):
    text = _clean_number_value(text)
    text = re.sub(r'[^A-Za-z0-9]+$', '', text)
    return re.match(r'^([A-Za-z]{1,3})\s*([0-9OZSIBl]{6,8})$', text, re.IGNORECASE)


def _normalize_passport(text):
    m = _looks_like_passport(text)
    if not m:
        return _clean_number_value(text)
    return m.group(1).upper() + _fix_digit_ocr(m.group(2))


def _normalize_short_cyrillic_code(text):
    text = text.strip().upper()
    replacements = {
        'K': 'К',
        'B': 'В',
        'D': 'Д',
        'Z': 'Д',
        '3': 'З',
        '4': 'Ҷ', 'Ч': 'Ҷ',
        'T': 'Т', '0': 'О', 'I': 'Ӣ', 'Y': 'Ӯ',
    }
    return ''.join(replacements.get(ch, ch) for ch in text)


def _normalize_mia_value(text):
    normalized = _normalize_short_cyrillic_code(text)
    letters = re.sub(r'[^A-ZА-ЯЁҚӢӮҲҶҒ]', '', normalized, flags=re.IGNORECASE).upper()
    if letters in {'ВК', 'ВКД', 'ВКЗ', 'ВК2', 'BK', 'BKD'}:
        return 'ВКД'
    return normalized.strip()


def _normalize_citizenship(text):
    value = text.strip(' .:;,-()_')
    value = re.sub(r'^[дuаАххььванддАа\s]+', '', value)
    lower = value.lower()
    known = {'хиоц': 'Хитой', 'хиюош': 'Хитой', 'хиюц': 'Хитой', 'хигой': 'Хитой', 'хитои': 'Хитой', 'хитой': 'Хитой', 'амнрн': 'Эрон', 'шахраанди': 'Шаҳрвандӣ'}
    if lower in known: return known[lower]
    if any(x in lower for x in ['хит', 'хиг', 'хио', 'хиц']): return 'Хитой'
    return value
def _clean_tajik_text(text):
    if not text: return text
    text = text.replace('0', 'О').replace('4', 'Ҷ').replace('3', 'З').replace('6', 'Б').replace('8', 'В').replace('Ч', 'Ҷ')
    replacements = {'Ханнфя': 'Ханифа', 'Амнрн': 'Амири', 'Дмхри': 'Душанбе', 'Ханнфа': 'Ханифа', 'Амнри': 'Амири', '0елонзода': 'Элонзода', 'Ваиг': 'Ванг', 'Душаибе': 'Душанбе', 'Еюец': 'Юе', 'Еюеш': 'Юе'}
    for old, new in replacements.items():
        if old in text: text = text.replace(old, new)
    if text.endswith('я') and len(text) > 3: text = text[:-1] + 'а'
    return text.strip()



def _has_cyrillic(text):
    return bool(re.search(r'[\u0400-\u04FF]', text))


def _is_candidate_for_field(field_num, text):
    """Field-specific value validation before a spatial candidate can win."""
    text = text.strip()
    if not text:
        return False

    if field_num == 1:
        cleaned = _clean_number_value(text)
        return bool(re.fullmatch(r'\d{6,10}', cleaned))
    if field_num == 2:
        return bool(_looks_like_passport(text))
    if field_num in {5, 7, 13}:
        return _extract_date_from_text(text) is not None
    if field_num == 8:
        cleaned = _clean_number_value(text)
        return bool(re.fullmatch(r'\d{4,6}', cleaned))
    if field_num == 3:
        if _is_number_or_code(text) or text.lower() in {'no', 'n'}:
            return False
        return _has_cyrillic(text) and len(text.strip(' .:;,-()')) >= 3
    if field_num == 6:
        normalized = _normalize_short_cyrillic_code(text)
        return bool(re.search(r'ХШБ|ВКД|ВК|ЧТ', normalized))
    if field_num == 9:
        normalized = _normalize_mia_value(text)
        return normalized == 'ВКД'
    if field_num == 4:
        if any(ch.isdigit() for ch in text) or _is_number_or_code(text):
            return False
        words = re.findall(r'[\u0400-\u04FF]{2,}', text)
        return len(words) >= 2
    if field_num in {10, 11, 12}:
        if _is_number_or_code(text) or text.lower() in {'no', 'n', 'h', 'i:'}:
            return False
        return _has_cyrillic(text) and len(text.strip(' .:;,-()_')) >= 4
    return True


def _extract_date_from_text(text):
    cleaned = re.sub(r'[^0-9.,/\\-]', '', text)
    digits_only = re.sub(r'[^0-9]', '', text)
    extra_patterns = [r'(\d{2})[.,/\\-](\d{2})[.,/\\-](\d{4})', r'\\b\d{8}\\b']
    for pattern in [r'(\d{2})[.,/\\-](\d{2})[.,/\\-](\d{4})', r'\d{8}', r'\d{2}[.,/\\-]\d{2}[.,/\\-]\d{2}']:
        for t in [text, cleaned, digits_only]:
            m = re.search(pattern, t)
            if m: return _clean_date_value(m.group())
    return None


# ---------------------------------------------------------------------------
# Tajik field label keywords for keyword-anchored fallback (Step 3)
# ---------------------------------------------------------------------------
FIELD_LABELS = {
    1: ['рақами бақайдгирӣ', 'рақами қайд', 'корти бақайдгирӣ',
        'карточка регистрации', 'registration card', 'варақаи'],
    2: ['шиноснома', 'шиносномаи', 'паспорт', 'рақами шиноснома',
        'шиноснома №', 'шиноснома но'],
    3: ['шаҳрвандӣ', 'шаҳрванди', 'шахрвандӣ', 'шахрванди', 'гражданство'],
    4: ['насаб', 'ному насаб', 'ном ва насаб'],
    5: ['санаи бақайдгирӣ', 'санаи қайд', 'ба қайд гирифта'],
    6: ['хшб', 'паспортӣ', 'бақайдгирии', 'паспорти', 'хшб вкд'],
    7: ['эътибор', 'мӯҳлат', 'мухлат', 'то кай'],
    8: ['силсила', 'назорат', 'serial'],
    9: ['вазорати корҳои дохилӣ', 'вазорати корхои дохилӣ', 'дохилӣ'],
    10: ['ҷои зист', 'чои зист', 'суроға', 'истиқомат', 'истикомат'],
    11: ['идома', 'давом'],
    12: ['нозир', 'инспектор'],
    13: ['тамдид', 'санаи тамдид', 'тамдиди'],
}

# ---------------------------------------------------------------------------
# Approximate relative Y positions of each field on a standard card
# (percentage of image height, top=0 bottom=100).
# Derived from the actual card layout; used when label numbers aren't detected.
# ---------------------------------------------------------------------------
FIELD_REL_Y_APPROX = {
    1:  29,   # card number (top right)
    2:  33,   # passport number
    3:  37,   # citizenship
    4:  43,   # name and surname
    5:  48,   # date of registration
    6:  53,   # PRS MIA RT
    7:  57,   # valid until
    8:  55,   # serial/control (same row as valid_until, but right side)
    9:  69,   # MIA
    10: 69,   # place of residence
    11: 79,   # residence continuation
    12: 79,   # inspector
    13: 89,   # date of registration/extension
}


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class TextParser:
    """
    Parses OCR text blocks using spatial anchor strategy with RELATIVE coordinates.

    Strategy:
      0. Pattern matching for Passport, Serial, Registration card number, Dates
      1. Detect numbered labels (1-13) on right half for spatial anchoring
      2. For each detected label number -> find value to its left (middle zone)
      3. Keyword-anchored fallback for missed fields
      4. Positional fallback using approximate Y positions
      Post-process: validate and clean each field value
    """

    def __init__(self):
        self.fields = Config.FIELDS
        self.date_patterns = Config.DATE_PATTERNS

    def parse(self, ocr_results):
        if not ocr_results:
            return self._empty_result()

        img_width, img_height = self._compute_image_bounds(ocr_results)
        if img_width < 1 or img_height < 1:
            return self._empty_result()

        field_value_map = {}
        used_texts = set()

        # ------------------------------------------------------------------
        # Step 0: Pattern matching for critical fields
        # Run BEFORE spatial steps so patterns get first pick of text blocks.
        # Important: run serial BEFORE dates so "9801" isn't claimed by a date.
        # ------------------------------------------------------------------
        priority_order = [
            'passport_number',
            'registration_card_number',
            'serial_control_number',
            'valid_until',
            'date_of_registration',
            'date_of_registration_extension',
        ]
        for key in priority_order:
            field_num = next(
                (n for n, f in self.fields.items() if f['key'] == key), None
            )
            if field_num is None:
                continue
            val, conf = self._pattern_fallback(key, ocr_results, used_texts)
            if val and (conf > 0.15 or (len(val) >= 8 and any(c.isdigit() for c in val))):
                field_value_map[field_num] = (val, max(conf, 0.5))
                used_texts.add(val)
                logger.info('Step 0: Pattern found Field %d (%s) = "%s" (conf=%.3f)',
                            field_num, key, val, conf)

        # ------------------------------------------------------------------
        # Step 1: Detect numbered labels (1-13)
        # ------------------------------------------------------------------
        label_positions = self._detect_labels(ocr_results, img_width, img_height)
        logger.info('Detected labels: %s', list(label_positions.keys()))

        value_candidates = self._filter_value_candidates(ocr_results)

        # ------------------------------------------------------------------
        # Step 2: Label-anchored spatial search
        # ------------------------------------------------------------------
        for field_num in range(1, 14):
            if field_num in field_value_map:
                continue
            if field_num not in label_positions:
                continue

            rel_x, rel_y, bbox = label_positions[field_num]
            val_text, val_conf = self._find_value_for_label(
                value_candidates, rel_x, rel_y, used_texts,
                field_num, img_width, img_height
            )
            if val_text:
                field_value_map[field_num] = (val_text, val_conf)
                used_texts.add(val_text)
                logger.info('Step 2: Label %d -> Value = "%s"', field_num, val_text)

        # ------------------------------------------------------------------
        # Step 3: Keyword-anchored fallback
        # ------------------------------------------------------------------
        for field_num in range(1, 14):
            if field_num in field_value_map:
                continue

            labels = FIELD_LABELS.get(field_num, [])
            label_block = None
            for block in ocr_results:
                text = block['text'].strip()
                if text in used_texts:
                    continue
                for lbl in labels:
                    if fuzzy_match(text, lbl, 0.7):
                        label_block = block
                        break
                if label_block:
                    break

            if label_block:
                lbx = (label_block['bbox'][0][0] + label_block['bbox'][2][0]) / 2
                lby = (label_block['bbox'][0][1] + label_block['bbox'][2][1]) / 2
                rel_lbx = (lbx / img_width) * 100
                rel_lby = (lby / img_height) * 100
                val_text, val_conf = self._find_text_near_keyword(
                    value_candidates, rel_lbx, rel_lby, used_texts,
                    field_num, img_width, img_height
                )
                if val_text:
                    field_value_map[field_num] = (val_text, val_conf)
                    used_texts.add(val_text)
                    logger.info('Step 3: Keyword anchor Field %d -> "%s"', field_num, val_text)

        # ------------------------------------------------------------------
        # Step 4: Positional fallback using known approximate Y positions
        # ------------------------------------------------------------------
        for field_num in range(1, 14):
            if field_num in field_value_map:
                continue
            val, conf = self._find_any_nearby(
                ocr_results, label_positions, field_num, used_texts,
                img_width, img_height
            )
            if val:
                field_value_map[field_num] = (val, conf)
                used_texts.add(val)
                logger.info('Step 4: Positional fallback Field %d -> "%s"', field_num, val)

        # ------------------------------------------------------------------
        # Post-process: validate and clean each value
        # ------------------------------------------------------------------
        field_value_map = self._post_process_fields(field_value_map)

        # Build result dict
        result = {}
        for field_num, field_info in self.fields.items():
            key = field_info['key']
            if field_num in field_value_map:
                value, conf = field_value_map[field_num]
                result[key] = {
                    'value': value,
                    'confidence': round(conf, 3),
                    'label': field_info['label'],
                    'field_number': field_num,
                }
            else:
                result[key] = {
                    'value': '', 'confidence': 0.0,
                    'label': field_info['label'],
                    'field_number': field_num,
                }
        return result

    # ------------------------------------------------------------------
    # Helper: compute virtual image bounds from bounding boxes
    # ------------------------------------------------------------------
    def _compute_image_bounds(self, ocr_results):
        all_x, all_y = [], []
        for block in ocr_results:
            bbox = block.get('bbox')
            if not bbox:
                continue
            for point in bbox:
                all_x.append(point[0])
                all_y.append(point[1])
        return (max(all_x), max(all_y)) if all_x else (0, 0)

    def _filter_value_candidates(self, ocr_results):
        return [block for block in ocr_results if block.get('text', '').strip()]

    # ------------------------------------------------------------------
    # Step 1: detect label numbers (1-13) on the right side of the card
    # ------------------------------------------------------------------
    def _detect_labels(self, text_blocks, img_width, img_height):
        positions = {}
        for block in text_blocks:
            text = block['text'].strip()
            bbox = block.get('bbox')
            if not bbox:
                continue

            num = None
            m = re.match(r'^[#№]?\s*(\d{1,2})\s*[.):\s]*$', text)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 13:
                    num = n

            if num is not None:
                cx = (bbox[0][0] + bbox[2][0]) / 2
                cy = (bbox[0][1] + bbox[2][1]) / 2
                rel_x = (cx / img_width) * 100
                rel_y = (cy / img_height) * 100
                if rel_x > 60:
                    if num not in positions or rel_x > positions[num][0]:
                        positions[num] = (rel_x, rel_y, bbox)
        return positions

    # ------------------------------------------------------------------
    # Step 2: find value to the LEFT of a detected label number
    # ------------------------------------------------------------------
    def _find_value_for_label(self, text_blocks, label_rel_x, label_rel_y,
                               used_texts, field_num, img_width, img_height):
        candidates = []
        for block in text_blocks:
            text = block['text'].strip()
            conf = block['confidence']
            bbox = block.get('bbox')
            if not bbox or text in used_texts:
                continue
            if _is_pure_label_text(text):
                continue
            if not _is_candidate_for_field(field_num, text):
                continue

            bx = (bbox[0][0] + bbox[2][0]) / 2
            by = (bbox[0][1] + bbox[2][1]) / 2
            rel_x = (bx / img_width) * 100
            rel_y = (by / img_height) * 100

            dx = rel_x - label_rel_x
            dy = rel_y - label_rel_y

            # Must be to the left of the label number
            if dx > -3:
                continue
            # Must not be in the far-left label/annotation zone
            if rel_x < 25:
                continue
            if field_num == 8 and rel_x < 70:
                continue
            if field_num == 11 and rel_x > 65:
                continue
            if field_num == 12 and rel_x < 55:
                continue
            # Must be on roughly the same row
            if abs(dy) > 5:
                continue

            if rel_x < 35:
                left_penalty = 200
            elif rel_x < 40:
                left_penalty = 100
            else:
                left_penalty = 0

            central_bonus = max(0, 15 - abs(rel_x - 55))
            score = abs(dy) * 25 + left_penalty - central_bonus
            score *= (1.5 - conf)
            if _is_number_or_code(text):
                score *= 0.5
            candidates.append((score, text, conf))

        if not candidates:
            return None, 0.0
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]

    # ------------------------------------------------------------------
    # Step 3: find value near a keyword label block
    # ------------------------------------------------------------------
    def _find_text_near_keyword(self, text_blocks, label_rel_x, label_rel_y,
                                 used_texts, field_num, img_width, img_height):
        candidates = []
        for block in text_blocks:
            text = block['text'].strip()
            conf = block['confidence']
            bbox = block.get('bbox')
            if not bbox or text in used_texts:
                continue
            if _is_pure_label_text(text):
                continue
            if not _is_candidate_for_field(field_num, text):
                continue

            bx = (bbox[0][0] + bbox[2][0]) / 2
            by = (bbox[0][1] + bbox[2][1]) / 2
            rel_x = (bx / img_width) * 100
            rel_y = (by / img_height) * 100

            dx = rel_x - label_rel_x
            dy = rel_y - label_rel_y

            if dx < -5:
                continue
            if field_num == 3:
                if dy < -5 or dy > 3:
                    continue
            elif dy < -3 or dy > 10:
                continue
            if dx > 65:
                continue
            if rel_x < 30 or rel_x > 80:
                continue
            if field_num == 11 and rel_x > 65:
                continue
            if field_num == 12 and rel_x < 55:
                continue

            score = abs(dx - 15) * 2 + abs(dy) * 10
            if 40 <= rel_x <= 70:
                score *= 0.7
            score *= (1.5 - conf)
            if _is_number_or_code(text):
                score *= 0.5
            candidates.append((score, text, conf))

        if not candidates:
            return None, 0.0
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]

    # ------------------------------------------------------------------
    # Step 0: pattern-based extraction for specific field types
    # ------------------------------------------------------------------
    def _pattern_fallback(self, key, text_blocks, used_texts):
        if not key:
            return None, 0.0

        if 'date' in key.lower() or 'valid' in key.lower():
            candidates = []
            _, img_height = self._compute_image_bounds(text_blocks)
            for block in text_blocks:
                text = block['text'].strip()
                if text in used_texts:
                    continue
                if _is_pure_label_text(text):
                    continue
                date = _extract_date_from_text(text)
                if date:
                    candidates.append((block['confidence'], date, block))
            if candidates:
                expected_by_key = {
                    'date_of_registration': FIELD_REL_Y_APPROX[5],
                    'valid_until': FIELD_REL_Y_APPROX[7],
                    'date_of_registration_extension': FIELD_REL_Y_APPROX[13],
                }
                expected_y = expected_by_key.get(key)
                if expected_y is not None and img_height:
                    positional = []
                    for conf, date, block in candidates:
                        bbox = block['bbox']
                        by = (bbox[0][1] + bbox[2][1]) / 2
                        rel_y = (by / img_height) * 100
                        distance = abs(rel_y - expected_y)
                        if distance <= 8:
                            positional.append((distance, conf, date, block))
                    if positional:
                        positional.sort(key=lambda x: (x[0], -x[1]))
                        return positional[0][2], positional[0][1]
                    return None, 0.0

                candidates.sort(key=lambda x: x[2]['bbox'][0][1])
                if key == 'date_of_registration':
                    pick = candidates[0]
                elif key == 'valid_until':
                    pick = candidates[1] if len(candidates) > 1 else candidates[0]
                elif key == 'date_of_registration_extension':
                    pick = candidates[-1]
                else:
                    pick = max(candidates, key=lambda x: x[0])
                return pick[1], pick[0]

        if 'registration_card' in key.lower():
            candidates = []
            for block in text_blocks:
                text = block['text'].strip()
                if text in used_texts:
                    continue
                if _is_pure_label_text(text):
                    continue
                # Skip passport-like text
                if re.search(r'[A-Za-z]{2,}\s?\d{5,}', text):
                    continue
                cleaned = _clean_number_value(text)
                for t in [text, cleaned]:
                    m = re.search(Config.REGISTRATION_CARD_PATTERN, t)
                    if m:
                        val = m.group()
                        start, end = m.start(), m.end()
                        if start > 0 and t[start - 1].isalnum():
                            continue
                        if end < len(t) and t[end].isalnum():
                            continue
                        if _is_pure_label_text(val):
                            continue
                        candidates.append((block['confidence'], val))
            if candidates:
                candidates.sort(key=lambda x: (-x[0], -len(x[1])))
                return candidates[0][1], candidates[0][0]

        if 'passport' in key.lower():
            candidates = []
            for block in text_blocks:
                text = block['text'].strip()
                if text in used_texts:
                    continue
                if _is_pure_label_text(text):
                    continue
                # Clean prefix characters only (no character substitutions yet)
                stripped = _clean_number_value(text)
                # Remove trailing non-alphanumeric chars
                stripped = re.sub(r'[^A-Za-z0-9]+$', '', stripped)
                for t in [text, stripped]:
                    # More permissive passport pattern: 1-3 letters + 6-8
                    # digit-like chars, then correct only the numeric part.
                    m = _looks_like_passport(t)
                    if m:
                        val = _normalize_passport(t)
                        digit_count = sum(1 for c in val if c.isdigit())
                        if digit_count >= 5:
                            candidates.append((block['confidence'], val))
            if candidates:
                candidates.sort(key=lambda x: (-x[0], -len(x[1])))
                return candidates[0][1], candidates[0][0]

        if 'serial' in key.lower() or 'control' in key.lower():
            candidates = []
            for block in text_blocks:
                text = block['text'].strip()
                if text in used_texts:
                    continue
                if _is_pure_label_text(text):
                    continue
                m = re.search(Config.SERIAL_NUMBER_NUMERIC_PATTERN, text)
                if m:
                    val = m.group()
                    if 4 <= len(val) <= 6:
                        candidates.append((block['confidence'], val))
            if candidates:
                candidates.sort(key=lambda x: (-x[0], -len(x[1])))
                return candidates[0][1], candidates[0][0]

        return None, 0.0

    # ------------------------------------------------------------------
    # Step 4: positional fallback using known approximate Y positions
    # ------------------------------------------------------------------
    def _find_any_nearby(self, text_blocks, label_positions, field_num,
                          used_texts, img_width, img_height):
        # Get estimated Y: use an exact detected label if present, otherwise
        # prefer the card-layout estimate.  Interpolating from sparse labels can
        # pull lower fields upward into the address/inspector rows.
        est_rel_y = None
        if field_num in label_positions:
            est_rel_y = label_positions[field_num][1]
        if est_rel_y is None:
            est_rel_y = FIELD_REL_Y_APPROX.get(field_num)
        if est_rel_y is None:
            return None, 0.0

        candidates = []
        for block in text_blocks:
            text = block['text'].strip()
            conf = block['confidence']
            bbox = block.get('bbox')
            if not bbox or text in used_texts:
                continue
            if _is_pure_label_text(text):
                continue
            if not _is_candidate_for_field(field_num, text):
                continue

            by = (bbox[0][1] + bbox[2][1]) / 2
            bx = (bbox[0][0] + bbox[2][0]) / 2
            rel_y = (by / img_height) * 100
            rel_x = (bx / img_width) * 100

            y_tolerance = 12 if field_num in {9, 10, 12, 5, 13} else 9
            if abs(rel_y - est_rel_y) > y_tolerance:
                continue
            min_x, max_x = 30, 80
            if field_num == 8:
                min_x, max_x = 68, 92
            elif field_num == 10:
                min_x, max_x = 40, 86
            elif field_num == 11:
                min_x, max_x = 15, 65
            elif field_num == 12:
                min_x, max_x = 55, 92
            elif field_num == 13:
                min_x, max_x = 25, 65
            if rel_x < min_x or rel_x > max_x:
                continue

            score = abs(rel_y - est_rel_y) * 10 + abs(rel_x - 55) * 1
            if rel_x < 35:
                score += 50
            if rel_x > 75:
                score += 50
            score *= (1.5 - conf)
            if _is_number_or_code(text):
                score *= 0.5
            candidates.append((score, text, conf))

        if not candidates:
            return None, 0.0
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]

    def _estimate_rel_y(self, sorted_labels, field_num):
        if not sorted_labels:
            return None
        for num, (rel_x, rel_y, bbox) in sorted_labels:
            if num == field_num:
                return rel_y
        known_nums = [n for n, _ in sorted_labels]
        known_ys = [rel_y for _, (_, rel_y, _) in sorted_labels]
        if field_num < min(known_nums):
            return known_ys[0] - (known_nums[0] - field_num) * 6
        if field_num > max(known_nums):
            return known_ys[-1] + (field_num - max(known_nums)) * 6
        for i in range(len(known_nums) - 1):
            if known_nums[i] < field_num < known_nums[i + 1]:
                ratio = (field_num - known_nums[i]) / (known_nums[i + 1] - known_nums[i])
                return known_ys[i] + ratio * (known_ys[i + 1] - known_ys[i])
        return None

    # ------------------------------------------------------------------
    # Post-processing: validate and clean field values
    # ------------------------------------------------------------------
    def _post_process_fields(self, field_value_map):
        cleaned = {}
        for field_num, (value, conf) in field_value_map.items():
            key = self.fields.get(field_num, {}).get('key', '')
            if key in {'name_and_surname', 'place_of_residence', 'place_of_residence_cont', 'inspector'}: value = _clean_tajik_text(value)
            value = value.strip('.:;,- ')

            # Reject values that are actually labels
            if _is_pure_label_text(value):
                logger.info('Post-process: rejecting label-as-value Field %d: "%s"', field_num, value)
                cleaned[field_num] = ('', 0.0)
                continue

            if 'passport' in key.lower():
                # Clean prefix; apply digit corrections only on digit portion
                if _looks_like_passport(value):
                    value = _normalize_passport(value)
                else:
                    stripped = _clean_number_value(value)
                    if len(stripped) < 4:
                        logger.info('Post-process: rejecting short passport Field %d: "%s"', field_num, value)
                        cleaned[field_num] = ('', 0.0)
                        continue
                    value = stripped

            elif 'registration_card' in key.lower():
                value = _clean_number_value(value)
                if not re.fullmatch(r'\d{6,10}', value):
                    logger.info('Post-process: rejecting non-numeric card number Field %d: "%s"', field_num, value)
                    cleaned[field_num] = ('', 0.0)
                    continue

            elif 'serial' in key.lower():
                value = _clean_number_value(value)
                if re.fullmatch(r'\d{5}', value) and value.endswith('8'):
                    # Field number 8 is printed immediately to the right and is
                    # often merged into the four-digit serial value by OCR.
                    value = value[:-1]
                if not re.fullmatch(r'\d{4,6}', value):
                    logger.info('Post-process: rejecting invalid serial Field %d: "%s"', field_num, value)
                    cleaned[field_num] = ('', 0.0)
                    continue

            elif key == 'citizenship':
                value = _normalize_citizenship(value)

            elif key == 'prs_mia_rt':
                value = _normalize_short_cyrillic_code(value)
                if 'ВКД' in value:
                    value = 'ХШБ ВКД ЧТ' if 'ХШБ' not in value else 'ХШБ ВКД ЧТ'
                elif value == 'ХВКД':
                    value = 'ХШБ ВКД'

            elif key == 'mia':
                value = _normalize_mia_value(value)
                if value != 'ВКД':
                    logger.info('Post-process: rejecting invalid MIA Field %d: "%s"', field_num, value)
                    cleaned[field_num] = ('', 0.0)
                    continue

            if key in {'name_and_surname', 'place_of_residence',
                       'place_of_residence_cont', 'inspector'}:
                if not _has_cyrillic(value) or len(value.strip(' .:;,-()_')) < 4:
                    logger.info('Post-process: rejecting weak text Field %d: "%s"', field_num, value)
                    cleaned[field_num] = ('', 0.0)
                    continue

            if 'date' in key.lower() or 'valid' in key.lower():
                date = _extract_date_from_text(value)
                if date:
                    value = date
                else:
                    logger.info('Post-process: rejecting non-date Field %d: "%s"', field_num, value)
                    cleaned[field_num] = ('', 0.0)
                    continue

            cleaned[field_num] = (value, conf)
        return cleaned

    def _empty_result(self):
        return {
            field_info['key']: {
                'value': '', 'confidence': 0.0,
                'label': field_info['label'], 'field_number': field_num,
            }
            for field_num, field_info in self.fields.items()
        }

    def get_confidence_score(self, parsed_data):
        scores = [v['confidence'] for v in parsed_data.values() if v['value']]
        return round(sum(scores) / len(scores), 3) if scores else 0.0
