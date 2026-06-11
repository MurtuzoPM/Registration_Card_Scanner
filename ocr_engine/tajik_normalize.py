"""
Tajik Cyrillic OCR normalization for Tajikistan registration cards.

Corrects common EasyOCR/Tesseract confusions (Latin letters read as Cyrillic,
missing Tajik letters ҳ/ҷ/ӯ/ғ/қ/ӣ, garbled labels) using a local lexicon —
no external API and no ML training step required.
"""
import json
import logging
import os
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_LEXICON_PATH = os.path.join(os.path.dirname(__file__), 'data', 'tajik_card_lexicon.json')

# Latin capitals often substituted for Cyrillic on red-printed cards
_LATIN_TO_CYR = str.maketrans({
    'A': 'А', 'B': 'В', 'C': 'С', 'D': 'Д', 'E': 'Е', 'F': 'Ф', 'G': 'Г',
    'H': 'Н', 'I': 'И', 'J': 'Ҷ', 'K': 'К', 'L': 'Л', 'M': 'М', 'N': 'Н',
    'O': 'О', 'P': 'Р', 'Q': 'Қ', 'R': 'Р', 'S': 'С', 'T': 'Т', 'U': 'У',
    'V': 'В', 'W': 'Ш', 'X': 'Х', 'Y': 'У', 'Z': 'З',
    'a': 'а', 'b': 'в', 'c': 'с', 'd': 'д', 'e': 'е', 'o': 'о', 'p': 'р',
    'k': 'к', 'x': 'х', 'y': 'у', 'h': 'н', 'm': 'м', 't': 'т',
})

# Tajik-specific: OCR often outputs Russian е instead of Tajik е/ӣ context
_CHAR_FIXES = {
    '\u0456': '\u0438',  # і → и
    '\u00ab': '', '\u00bb': '', '\u201c': '', '\u201d': '',
    '\u2018': "'", '\u2019': "'",
}

# Ordered regex phrase fixes (from real card OCR in uploads/)
_PHRASE_REGEX = [
    (r'чумхур[иiл]?\s+точикистон', 'Ҷумҳурии Тоҷикистон'),
    (r'точик[иi]?нстон', 'Тоҷикистон'),
    (r'точикистон', 'Тоҷикистон'),
    (r'шахрван[дл][иi]?\b', 'шаҳрванди'),
    (r'шахрванд[иi]\b', 'шаҳрванди'),
    (r'шиноснома[иi]?\b', 'шиносномаи'),
    (r'варточка\s+регистрации', 'корти бақайдгирӣ'),
    (r'карточка\s+регистрации', 'корти бақайдгирӣ'),
    (r'регистраци[иi]\b', 'бақайдгирӣ'),
    (r'бакайдгири', 'бақайдгирӣ'),
    (r'бакаидгир', 'бақайдгир'),
    (r'хшб\s*в[kк][дd]\s*г?м?хурни', 'ХШБ ВКД ЧТ'),
    (r'хшб\s*в[kк][дd]\s*ч?т?', 'ХШБ ВКД ЧТ'),
    (r'хшб\s*в[kк]\b', 'ХШБ ВКД'),
    (r'вазорати\s+корхои\s+дохил', 'Вазорати корҳои дохилӣ'),
    (r'охилии', 'дохилӣ'),
    (r'греза', 'гуреза'),
    (r'\bкуч\b', 'кӯча'),
    (r'\bхуч\b', 'кӯча'),
    (r'\bжуч\b', 'кӯча'),
    (r'паспорти\s+бақайдгирии', 'паспортӣ-бақайдгирии'),
    (r'gistation', 'registration'),
    (r'registration\s+card', 'корти бақайдгирӣ'),
]

# Tokens that are almost always OCR garbage on these cards (Latin noise)
_NOISE_EXACT = frozenset({
    'rrr?', 'trt?', 'tt', 'ed', 'sd', 'fla', 'ees', 'uz', 'pn', 'te', 'vo',
    'ba', 'och', 'se.', 'oo', 'card', 'az', 're', 'de', 'jw', 'wt', 'sn',
    'wer', 'ote', 'para', 'sen', 'sk', 'ak', 'cp', 'ses', 'prom', 'ads',
    'fn', 'pb', 'ny', 'st', 'nts', 'srs', 'yt?', 'ka', 'bl', 'fa', 'ww',
    'me', 'or', 've', 'zt', 'ma,', 'ms:', 'oa', 'ay', 'ah', 'acm', 'sf',
    'gee', 'one', 'bea', 'mt', 'ct', 'bs', 'bi', 'an', 'ra,', 'da,',
})


def _load_lexicon():
    try:
        with open(_LEXICON_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        logger.warning('Could not load Tajik lexicon: %s', exc)
        return {}


_LEXICON = _load_lexicon()
_PHRASE_MAP = {
    k.lower(): v for k, v in _LEXICON.get('phrase_corrections', {}).items()
}


def cyrillic_ratio(text):
    if not text:
        return 0.0
    cyr = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
    return cyr / len(text)


def is_ocr_noise(text, confidence=1.0):
    """Drop obvious non-field OCR fragments before parsing."""
    t = text.strip()
    if not t:
        return True
    low = t.lower().strip('.,;:!?\"\'()[]{}')

    if low in _NOISE_EXACT:
        return True

    # Pure punctuation / symbols
    if re.fullmatch(r'[\W_]{1,4}', t):
        return True

    # Short Latin garbage (Trt?, rrr?, ed)
    if len(t) <= 5 and re.fullmatch(r'[A-Za-z\?`.,;:!@#$%^&*+\-\\/|]{1,5}', t):
        return True

    # Very short, low confidence, not a number
    if len(t) <= 2 and confidence < 0.5 and not t.isdigit():
        return True

    # Mostly Latin in a Tajik card context (except passport codes)
    if len(t) >= 3 and cyrillic_ratio(t) < 0.15:
        if not re.search(r'^[A-Z]{1,3}\d{5,}', t.replace(' ', '')):
            if not re.fullmatch(r'\d{4,10}', t):
                if confidence < 0.55:
                    return True

    return False


def fix_latin_in_cyrillic_word(word):
    """Convert Latin homoglyphs to Cyrillic when word looks like Cyrillic text."""
    if not word or cyrillic_ratio(word) < 0.25:
        return word
    letters = sum(c.isalpha() for c in word)
    latin = sum(1 for c in word if 'A' <= c <= 'Z' or 'a' <= c <= 'z')
    if letters and latin / letters > 0.4:
        return word.translate(_LATIN_TO_CYR)
    # Mixed: fix only isolated Latin capitals inside Cyrillic words
    out = []
    for ch in word:
        if 'A' <= ch <= 'Z' and cyrillic_ratio(word) > 0.3:
            out.append(_LATIN_TO_CYR.get(ch, ch))
        else:
            out.append(ch)
    return ''.join(out)


def normalize_text(text):
    """Apply Tajik-aware corrections to a single OCR string."""
    if not text:
        return ''

    for old, new in _CHAR_FIXES.items():
        text = text.replace(old, new)

    text = re.sub(r'\s+', ' ', text.strip())

    low = text.lower()
    if low in _PHRASE_MAP:
        return _PHRASE_MAP[low]

    # Full-string phrase fixes before per-word Latin mapping (avoids double letters)
    for pattern, repl in _PHRASE_REGEX:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Substring lexicon only for long phrases (avoid "хшб вк" inside "хшб вкд чт")
    for src, dst in sorted(_PHRASE_MAP.items(), key=lambda x: -len(x[0])):
        if len(src) < 12:
            continue
        if src in text.lower():
            text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)

    # Word-level Latin→Cyrillic only when phrase rules did not already rewrite
    words = text.split()
    fixed_words = []
    for w in words:
        w_low = w.lower().strip('.,;:!?()')
        if w_low in _PHRASE_MAP:
            fixed_words.append(_PHRASE_MAP[w_low])
        elif re.search(r'[A-Za-z]', w) and cyrillic_ratio(w) >= 0.2:
            fixed_words.append(fix_latin_in_cyrillic_word(w))
        else:
            fixed_words.append(w)
    text = ' '.join(fixed_words)

    return text.strip()


def normalize_ocr_block(block):
    """Normalize one OCR result dict; may return None if block is noise."""
    text = block.get('text', '')
    conf = float(block.get('confidence', 0))
    cleaned = normalize_text(text)
    if not cleaned or is_ocr_noise(cleaned, conf):
        return None
    out = dict(block)
    out['text'] = cleaned
    if cleaned != text:
        out['raw_text'] = text
    return out


def normalize_ocr_results(ocr_results):
    """Filter noise and correct Tajik text on all OCR blocks."""
    out = []
    dropped = 0
    for block in ocr_results:
        normalized = normalize_ocr_block(block)
        if normalized:
            out.append(normalized)
        else:
            dropped += 1
    if dropped:
        logger.info('Tajik normalize: kept %d blocks, dropped %d noise',
                    len(out), dropped)
    return out


def fuzzy_match_lexicon(text, candidates, threshold=0.78):
    """Match text against a list of known values (countries, cities)."""
    low = text.lower().strip(' .:;,-()')
    for cand in candidates:
        if cand.lower() in low or low in cand.lower():
            return cand
    best = None
    best_score = threshold
    for cand in candidates:
        score = SequenceMatcher(None, low, cand.lower()).ratio()
        if score > best_score:
            best_score = score
            best = cand
    return best


def get_field_values(key):
    """Return known values list from lexicon for a field category."""
    fv = _LEXICON.get('field_values', {})
    return fv.get(key, [])