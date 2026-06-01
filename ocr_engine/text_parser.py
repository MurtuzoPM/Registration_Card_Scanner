import re
import logging
from difflib import SequenceMatcher
from config import Config

logger = logging.getLogger(__name__)


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
    return bool(re.search(r'\d{2}[./\-]\d{2}[./\-]\d{2,4}', text.strip()))


def _is_number_or_code(text):
    """Check if text looks like a number, passport code, or serial number."""
    text = text.strip()
    # Capped at 9 chars to avoid matching long label misreads (e.g. "REG15TRAT10")
    if re.match(r'^[A-Z0-9]{1,3}\s?[A-Z0-9]{6,8}$', text, re.IGNORECASE):
        return True
    if re.match(r'^\d{3,}$', text):
        return True
    if re.match(r'^[A-Z0-9\-]{5,9}$', text, re.IGNORECASE):
        return True
    if re.match(r'^№?\s?\d+$', text):
        return True
    return False


def _is_pure_label_text(text):
    """
    Check if text is likely a field description/label rather than a value.
    Highly aggressive for specific registration card keywords.
    """
    text_stripped = text.strip()
    if not text_stripped:
        return True

    # Check for dates first (always value)
    if _is_date_text(text_stripped):
        return False
    
    is_code = _is_number_or_code(text_stripped)
    words = text_stripped.split()
    text_lower = text_stripped.lower()
    
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

    # Extended label keywords (common OCR misread variants)
    # These are generated from base keywords with common OCR character confusions
    extended_label_keywords = base_label_keywords + [
        # Misreads of бақайдгирӣ
        'бакаидгири', 'бакайдгири', 'бакаидгирӣ', 'бакаидгирий',
        'бакайдгирий',
        # Misreads of вазорати
        'изорати', 'взорати', 'возорати',
        # Misreads of дохилӣ
        'дозилин', 'дозищив', 'дохилин', 'дохили',
        # Misreads of чумхурии
        'цумхурии', 'цумхурни', 'циумхурни',
        # Misreads of шаҳрвандӣ
        'шахрванди', 'шахрапди', 'шахраанди',
        # Misreads of шиноснома
        'шиносномаи',
        # Misreads of эътибор
        'эьтибор', 'эъти6ор',
        # Misreads of тамдид
        'тадид', 'тамди',
        # Common English label misreads
        'kecistration', 'registrat', 'reistratio',
    ]

    # Short label fragments that should never be values
    short_label_fragments = {'no', 'n', 'h', 'i', 'i:', 'т', 'фм', 'тм', 'tм', 'вк', 'вкz'}
    if text_lower.strip('.:;,- ') in short_label_fragments:
        return True

    # Label numbers 1-13 on the right side — these are markers, not values
    if text_stripped.isdigit():
        n = int(text_stripped)
        if 1 <= n <= 13:
            return True

    # If text is very short (1-2 chars) and not a clear code/number, it's noise
    if len(text_stripped) <= 2 and not text_stripped.isdigit():
        return True

    # Check for exact or prefix match with any keyword
    for kw in extended_label_keywords:
        kw_lower = kw.lower()
        if text_lower == kw_lower:
            return True
        if text_lower.startswith(kw_lower):
            # Only match prefix if the text isn't much longer (meaning it's likely the same word)
            if len(text_lower) <= len(kw_lower) + 3:
                return True

    # Fuzzy match single words against known keywords (full word comparison)
    if len(words) == 1:
        for kw in base_label_keywords:
            if len(kw) > 3 and SequenceMatcher(None, text_lower, kw.lower()).ratio() >= 0.75:
                return True

    # Fuzzy match multi-word texts against known multi-word phrases
    if len(words) >= 2:
        for kw in base_label_keywords:
            if ' ' in kw:
                kw_parts = kw.split()
                # Check if the text matches the keyword roughly
                if SequenceMatcher(None, text_lower, kw.lower()).ratio() >= 0.7:
                    return True
                # Check word by word
                if len(words) == len(kw_parts):
                    matches = 0
                    for i, (tw, kw_w) in enumerate(zip(words, kw_parts)):
                        if SequenceMatcher(None, tw, kw_w).ratio() >= 0.7:
                            matches += 1
                    if matches >= len(kw_parts) - 1:
                        return True

    # If it starts with a known label keyword and it's short, it's likely a label
    if len(words) <= 2 and any(text_lower.startswith(k) for k in extended_label_keywords):
        return True

    # Flag distinctly descriptive phrases
    strong_label_phrases = [
        'рақами корти бақайдгирӣ', 'рақами шиноснома', 'ном ва насаб',
        'санаи бақайдгирӣ', 'то кай эътибор', 'ҷои зист', 'чои зист',
        'вазорати корҳои', 'вазидентаи корхои', 'паспортӣ-бақайдгирии',
        'паспортӣ бақайдгирии', 'бақайдгирӣ ё тамдид', 'корти бақайдгирӣ',
        'карточка регистрации', 'registration card', 'варақаи бақайдгирӣ',
        'ба қайд гирифта шуд', 'тамдид карда шуд', 'чои истикомат',
        'идомаи ҷои зист', 'вазорати корхои дохили', 'чумхурии точикистон',
        'барои истифода', 'кабулшуда', 'ташкилот', 'шахси', 'кабулкунанда',
        'ба кайд гирифта', 'ба қайд гирифта',
    ]
    for phrase in strong_label_phrases:
        if phrase in text_lower:
            return True
    
    # Also check if it contains multiple label keywords
    match_count = sum(1 for k in base_label_keywords if k in text_lower)
    if match_count >= 2 and len(words) < 6:
        return True

    # If text starts with "(" or ")", it's likely an annotation/label, not a value
    if text_stripped.startswith('(') or text_stripped.startswith(')'):
        return True

    # Even code-like text might be a label misread — check fuzzy match
    if is_code:
        for kw in base_label_keywords:
            if len(kw) > 3:
                # Clean digits from text and compare
                cleaned = re.sub(r'[0-9]', '', text_lower)
                if len(cleaned) > 3 and SequenceMatcher(None, cleaned, kw.lower()).ratio() >= 0.6:
                    return True
                # Check individual words against cleaned words
                for tw in text_lower.split():
                    tw_clean = re.sub(r'[0-9]', '', tw)
                    if len(tw_clean) > 3 and SequenceMatcher(None, tw_clean, kw.lower()).ratio() >= 0.65:
                        return True

    # Text that is mostly uppercase and shares high similarity with any label keyword
    if text_stripped.isupper() and len(text_stripped) > 3:
        for kw in base_label_keywords:
            if len(kw) > 3:
                if SequenceMatcher(None, text_lower, kw.lower()).ratio() >= 0.6:
                    return True
                # Check individual words
                text_words = text_lower.split()
                kw_words = kw.split()
                for tw in text_words:
                    for kw_w in kw_words:
                        if len(tw) > 3 and len(kw_w) > 3:
                            if SequenceMatcher(None, tw, kw_w).ratio() >= 0.7:
                                return True
                # Also check: does the text clean up to something very close to a label?
                cleaned = re.sub(r'[^a-z\u0400-\u04FF]', '', text_lower)
                if len(cleaned) > 3 and SequenceMatcher(None, cleaned, kw.lower()).ratio() >= 0.7:
                    return True

    return False


def _clean_date_value(text):
    """Normalize date format to DD.MM.YYYY."""
    text = text.strip()
    m = re.search(r'(\d{2})[./\-](\d{2})[./\-](\d{4})', text)
    if m:
        return f'{m.group(1)}.{m.group(2)}.{m.group(3)}'
    m = re.search(r'(\d{2})[./\-](\d{2})[./\-](\d{2})\b', text)
    if m:
        year = int(m.group(3))
        full_year = f'20{m.group(3)}' if year < 80 else f'19{m.group(3)}'
        return f'{m.group(1)}.{m.group(2)}.{full_year}'
    return text


def _clean_number_value(text):
    """Strip common prefixes from number fields (№, No, N)."""
    text = text.strip()
    text = re.sub(r'^[№NnNo\.:\s]+', '', text).strip()
    # Strip trailing dash, underscore, and non-alphanumeric chars
    text = re.sub(r'[-_\'"\s]+$', '', text).strip()
    # Correct common OCR misreads
    text = text.upper().replace('O', '0').replace('Z', '7').replace('S', '5').replace('I', '1').replace('B', '8')
    return text


def _extract_date_from_text(text):
    """Try to extract a date from a text string."""
    for pattern in Config.DATE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return _clean_date_value(m.group())
    return None


# Tajik field labels for keyword-based fallback
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


class TextParser:
    """
    Parses OCR text blocks using spatial anchor strategy with RELATIVE coordinates.

    Strategy:
      1. Pattern match critical fields (Passport, Serial, Dates)
      2. Detect numbered labels (1-13) on right half
      3. Search for values in the MIDDLE zone (rel_x 35-80%)
      4. Fallback to keywords and position estimates
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

        # Step 0: Pattern matching for critical fields (Passport, Serial, Dates, Registration Card)
        # This is more reliable than spatial mapping if patterns are unique
        for field_num in range(1, 14):
            key = self.fields.get(field_num, {}).get('key', '')
            if any(k in key.lower() for k in ['registration_card', 'passport', 'serial', 'date', 'valid']):
                val, conf = self._pattern_fallback(key, ocr_results, used_texts)
                # Lower threshold for pattern matches — pattern is more reliable than confidence
                if val and conf > 0.2:
                    field_value_map[field_num] = (val, max(conf, 0.5))
                    used_texts.add(val)
                    logger.info('Step 0: Pattern found for Field %d = "%s" (conf=%.3f)', field_num, val, conf)

        # Step 1: Detect numbered labels (1-13)
        label_positions = self._detect_labels(ocr_results, img_width, img_height)
        logger.info('Detected labels: %s', list(label_positions.keys()))

        value_candidates = self._filter_value_candidates(ocr_results)

        # Step 2: Found label numbers → find values to the left (middle zone)
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
                logger.info('Step 2: Label %d found Value = "%s"', field_num, val_text)

        # Step 3: Keyword-anchored fallback
        for field_num in range(1, 14):
            if field_num in field_value_map:
                continue

            labels = FIELD_LABELS.get(field_num, [])
            label_block = None
            for block in ocr_results:
                text = block['text'].strip()
                if text in used_texts: continue
                for lbl in labels:
                    if fuzzy_match(text, lbl, 0.7):
                        label_block = block
                        break
                if label_block: break

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
                    logger.info('Step 3: Keyword found for Field %d = "%s"', field_num, val_text)

        # Step 4: Final positional fallback
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
                logger.info('Step 4: Positional fallback for Field %d = "%s"', field_num, val)

        # Post-process
        field_value_map = self._post_process_fields(field_value_map)

        # Build result
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
                    'label': field_info['label'], 'field_number': field_num,
                }
        return result

    def _compute_image_bounds(self, ocr_results):
        all_x = []
        all_y = []
        for block in ocr_results:
            bbox = block.get('bbox')
            if not bbox: continue
            for point in bbox:
                all_x.append(point[0])
                all_y.append(point[1])
        return (max(all_x), max(all_y)) if all_x else (0, 0)

    def _filter_value_candidates(self, ocr_results):
        return [block for block in ocr_results if block.get('text', '').strip()]

    def _detect_labels(self, text_blocks, img_width, img_height):
        positions = {}
        for block in text_blocks:
            text = block['text'].strip()
            bbox = block.get('bbox')
            if not bbox: continue

            # Match pure numbers 1-13 on the right half
            num = None
            m = re.match(r'^[#№]?\s*(\d{1,2})\s*[.):\s]*$', text)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 13: num = n

            if num is not None:
                cx = (bbox[0][0] + bbox[2][0]) / 2
                cy = (bbox[0][1] + bbox[2][1]) / 2
                rel_x = (cx / img_width) * 100
                rel_y = (cy / img_height) * 100
                if rel_x > 60: # Must be on the right side
                    if num not in positions or rel_x > positions[num][0]:
                        positions[num] = (rel_x, rel_y, bbox)
        return positions

    def _find_value_for_label(self, text_blocks, label_rel_x, label_rel_y,
                               used_texts, field_num, img_width, img_height):
        candidates = []
        for block in text_blocks:
            text = block['text'].strip()
            conf = block['confidence']
            bbox = block.get('bbox')
            if not bbox or text in used_texts: continue
            if _is_pure_label_text(text): continue

            bx = (bbox[0][0] + bbox[2][0]) / 2
            by = (bbox[0][1] + bbox[2][1]) / 2
            rel_x = (bx / img_width) * 100
            rel_y = (by / img_height) * 100

            dx = rel_x - label_rel_x
            dy = rel_y - label_rel_y

            # Must be to the LEFT of the label number (which is on the right side)
            if dx > -3: continue
            # Must not be too far left (label area is typically rel_x < 35%)
            if rel_x < 25: continue
            # Must be roughly same row
            if abs(dy) > 5: continue

            # Strongly penalize far-left text, reward middle zone (rel_x 40-70%)
            if rel_x < 35:
                left_penalty = 200
            elif rel_x < 40:
                left_penalty = 100
            else:
                left_penalty = 0

            central_bonus = max(0, 15 - abs(rel_x - 55))

            score = abs(dy) * 25 + left_penalty - central_bonus
            # Penalize lower confidence
            score *= (1.5 - conf)
            # Bonus for code-like text (passport, serial numbers)
            if _is_number_or_code(text):
                score *= 0.5
            candidates.append((score, text, conf))

        if not candidates: return None, 0.0
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]

    def _find_text_near_keyword(self, text_blocks, label_rel_x, label_rel_y,
                                 used_texts, field_num, img_width, img_height):
        candidates = []
        for block in text_blocks:
            text = block['text'].strip()
            conf = block['confidence']
            bbox = block.get('bbox')
            if not bbox or text in used_texts: continue
            if _is_pure_label_text(text): continue

            bx = (bbox[0][0] + bbox[2][0]) / 2
            by = (bbox[0][1] + bbox[2][1]) / 2
            rel_x = (bx / img_width) * 100
            rel_y = (by / img_height) * 100

            dx = rel_x - label_rel_x
            dy = rel_y - label_rel_y

            # Must be to the right of the label keyword
            if dx < -5: continue
            # Must be on same row or slightly below
            if dy < -3 or dy > 10: continue
            if dx > 65: continue
            # Focus on middle value zone
            if rel_x < 30 or rel_x > 80: continue

            score = abs(dx - 15) * 2 + abs(dy) * 10
            # Bonus for middle zone
            if 40 <= rel_x <= 70:
                score *= 0.7
            score *= (1.5 - conf)
            if _is_number_or_code(text):
                score *= 0.5
            candidates.append((score, text, conf))

        if not candidates: return None, 0.0
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]

    def _pattern_fallback(self, key, text_blocks, used_texts):
        if not key: return None, 0.0

        if 'date' in key.lower():
            candidates = []
            for block in text_blocks:
                text = block['text'].strip()
                if text in used_texts: continue
                if _is_pure_label_text(text): continue
                date = _extract_date_from_text(text)
                if date: candidates.append((block['confidence'], date))
            if candidates:
                # Sort by confidence (higher is better)
                candidates.sort(key=lambda x: -x[0])
                return candidates[0][1], candidates[0][0]

        if 'registration_card' in key.lower():
            candidates = []
            for block in text_blocks:
                text = block['text'].strip()
                if text in used_texts: continue
                if _is_pure_label_text(text): continue
                cleaned = _clean_number_value(text)
                # Skip if this text block looks like a passport number (digits + letters prefix)
                if re.search(r'[A-Za-z]{2,}\s?\d{5,}', cleaned):
                    continue
                for t in [text, cleaned]:
                    m = re.search(Config.REGISTRATION_CARD_PATTERN, t)
                    if m:
                        val = m.group()
                        if _is_pure_label_text(val): continue
                        # Reject if the match is a substring of a larger alphanumeric token
                        # that also has letters (e.g. "567165" inside "EC567165")
                        start = m.start()
                        end = m.end()
                        if start > 0 and t[start-1].isalnum():
                            continue
                        if end < len(t) and t[end].isalnum():
                            continue
                        candidates.append((block['confidence'], val))
            if candidates:
                candidates.sort(key=lambda x: (-x[0], -len(x[1])))
                return candidates[0][1], candidates[0][0]

        if 'passport' in key.lower():
            candidates = []
            for block in text_blocks:
                text = block['text'].strip()
                if text in used_texts: continue
                # Skip text that looks like a label (e.g. "REGISTRATIO" from "REGISTRATION")
                if _is_pure_label_text(text):
                    continue
                # Strip trailing non-alphanumeric chars before matching
                stripped = re.sub(r'[^A-Za-z0-9\u0400-\u04FF]+$', '', text)
                for t in [text, stripped, _clean_number_value(text)]:
                    m = re.search(Config.PASSPORT_NUMBER_PATTERN, t, re.IGNORECASE)
                    if m:
                        val = re.sub(r'[^A-Z0-9]', '', m.group().upper())
                        if len(val) >= 7:
                            # Reject if value looks like a word (mostly letters, few digits)
                            digit_count = sum(1 for c in val if c.isdigit())
                            if digit_count < 2:
                                continue
                            candidates.append((block['confidence'], val))
            if candidates:
                candidates.sort(key=lambda x: (-x[0], -len(x[1])))
                return candidates[0][1], candidates[0][0]

        if 'serial' in key.lower() or 'control' in key.lower():
            candidates = []
            for block in text_blocks:
                text = block['text'].strip()
                if text in used_texts: continue
                # Skip text that looks like a label
                if _is_pure_label_text(text):
                    continue
                # Try to extract 4-6 digit numeric serial codes
                m = re.search(Config.SERIAL_NUMBER_NUMERIC_PATTERN, text)
                if m:
                    val = m.group()
                    if 4 <= len(val) <= 6:
                        candidates.append((block['confidence'], val))
            if candidates:
                candidates.sort(key=lambda x: (-x[0], -len(x[1])))
                return candidates[0][1], candidates[0][0]

        return None, 0.0

    def _find_any_nearby(self, text_blocks, label_positions, field_num,
                          used_texts, img_width, img_height):
        sorted_labels = sorted(label_positions.items())
        est_rel_y = self._estimate_rel_y(sorted_labels, field_num)
        if est_rel_y is None: return None, 0.0

        candidates = []
        for block in text_blocks:
            text = block['text'].strip()
            conf = block['confidence']
            bbox = block.get('bbox')
            if not bbox or text in used_texts: continue
            if _is_pure_label_text(text): continue
            
            by = (bbox[0][1] + bbox[2][1]) / 2
            bx = (bbox[0][0] + bbox[2][0]) / 2
            rel_y = (by / img_height) * 100
            rel_x = (bx / img_width) * 100

            if abs(rel_y - est_rel_y) > 7: continue
            # Only accept text in the middle zone (not left labels, not right numbers)
            if rel_x < 30 or rel_x > 80: continue
            # Heavily favor middle zone
            score = abs(rel_y - est_rel_y) * 10 + abs(rel_x - 55) * 1
            if rel_x < 35: score += 50
            if rel_x > 75: score += 50
            
            score *= (1.5 - conf)
            if _is_number_or_code(text):
                score *= 0.5
            candidates.append((score, text, conf))

        if not candidates: return None, 0.0
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]

    def _estimate_rel_y(self, sorted_labels, field_num):
        if not sorted_labels: return None
        for num, (rel_x, rel_y, bbox) in sorted_labels:
            if num == field_num: return rel_y
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

    def _post_process_fields(self, field_value_map):
        cleaned = {}
        for field_num, (value, conf) in field_value_map.items():
            value = value.strip('.:;,- ')
            key = self.fields.get(field_num, {}).get('key', '')

            # Reject values that are actually known labels
            if _is_pure_label_text(value):
                logger.info('Post-process: rejecting label-as-value for Field %d: "%s"', field_num, value)
                cleaned[field_num] = ('', 0.0)
                continue

            if any(k in key.lower() for k in ['passport', 'registration', 'serial']):
                value = _clean_number_value(value)
                # Reject if cleaning removed everything or result is too short
                if len(value) < 4:
                    logger.info('Post-process: rejecting short code for Field %d: "%s"', field_num, value)
                    cleaned[field_num] = ('', 0.0)
                    continue

            if 'date' in key.lower():
                date = _extract_date_from_text(value)
                if date:
                    value = date
                else:
                    # Reject non-date values for date fields
                    logger.info('Post-process: rejecting non-date for Field %d: "%s"', field_num, value)
                    cleaned[field_num] = ('', 0.0)
                    continue

            cleaned[field_num] = (value, conf)
        return cleaned

    def _empty_result(self):
        result = {}
        for field_num, field_info in self.fields.items():
            result[field_info['key']] = {
                'value': '', 'confidence': 0.0,
                'label': field_info['label'], 'field_number': field_num,
            }
        return result

    def get_confidence_score(self, parsed_data):
        scores = [v['confidence'] for v in parsed_data.values() if v['value']]
        return round(sum(scores) / len(scores), 3) if scores else 0.0
