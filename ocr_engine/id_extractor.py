"""
Field 1 (registration card #) vs Field 2 (passport #) — layout-aware extraction.

On Tajik registration cards (ground-truth example):
  - Registration card number: 6–10 digits (e.g. 1233509), near «рақами корти бақайдгирӣ»
  - Passport number: № + digits (e.g. №0011801) or letter+digits (e.g. EC5671065),
    on the «шиносномаи» row — OCR often drops «№» and reads only 0011801
"""
import re
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

PASSPORT_ALPHA_RE = re.compile(
    r'^([A-Z]{1,3})\s*([0-9OIZSB]{6,8})$', re.IGNORECASE
)
PURE_DIGITS_RE = re.compile(r'^\d{6,10}$')
PASSPORT_EMBEDDED_RE = re.compile(
    r'([A-Z]{1,3})\s*([0-9OIZSB]{6,8})', re.IGNORECASE
)
PASSPORT_NUM_PREFIX_RE = re.compile(
    r'^[№NnNo.\s]*(\d{6,8})$', re.IGNORECASE
)

PASSPORT_LABEL_KEYWORDS = [
    'шиноснома', 'шиносномаи', 'шиносном', 'паспорт', 'рақами шиноснома',
]
REGISTRATION_LABEL_KEYWORDS = [
    'бақайдгирӣ', 'бақайдгири', 'рақами корти', 'корти бақайд',
    'registration card', 'рақами бақайд', 'варақаи',
]


def _center(bb):
    xs = [p[0] for p in bb]
    ys = [p[1] for p in bb]
    return sum(xs) / 4.0, sum(ys) / 4.0


def _bounds(blocks):
    xs, ys = [], []
    for b in blocks:
        for p in b['bbox']:
            xs.append(p[0])
            ys.append(p[1])
    if not xs:
        return 1, 1, 0, 0
    return max(xs), max(ys), min(xs), min(ys)


def _rel_y(bb, blocks):
    _, max_y, _, min_y = _bounds(blocks)
    cy = _center(bb)[1]
    span = max(max_y - min_y, 1)
    return (cy - min_y) / span * 100.0


def _rel_x(bb, blocks):
    max_x, _, min_x, _ = _bounds(blocks)
    cx = _center(bb)[0]
    span = max(max_x - min_x, 1)
    return (cx - min_x) / span * 100.0


def _normalize_passport_alpha(prefix, digits):
    digits = (
        digits.upper()
        .replace('O', '0').replace('I', '1').replace('Z', '7')
        .replace('S', '5').replace('B', '8').replace('L', '1')
    )
    return prefix.upper() + digits


def format_passport_number(digits):
    """Tajik internal passport: № + zero-padded digits (e.g. №0011801)."""
    digits = re.sub(r'[^0-9]', '', digits)
    if not digits or not PURE_DIGITS_RE.match(digits):
        return None
    return f'№{digits}'


def _parse_passport_numeric(text, card_number=None):
    """№0011801 or bare 0011801 (passport row / № prefix)."""
    raw = text.strip()
    has_num_sign = bool(re.match(r'^[№NnNo]', raw))
    m = PASSPORT_NUM_PREFIX_RE.match(raw)
    if not m:
        compact = re.sub(r'[^0-9№NnNo]', '', raw)
        m = PASSPORT_NUM_PREFIX_RE.match(compact)
    if not m:
        return None
    digits = m.group(1)
    if card_number and digits == card_number:
        return None
    if not PURE_DIGITS_RE.match(digits):
        return None
    # Bare 7-digit block without №: treat as passport only if № was present
    # or value looks like internal passport (leading zeros), not registration.
    if not has_num_sign and not digits.startswith('00'):
        return None
    return format_passport_number(digits)


def _parse_passport_alpha(text, card_number=None):
    text = re.sub(r'[^A-Za-z0-9]+$', '', text.strip())
    text = re.sub(r'^[№NnNo.\s]+', '', text)
    if PURE_DIGITS_RE.match(text):
        return None
    m = PASSPORT_ALPHA_RE.match(text.upper())
    if m:
        prefix, digits = m.group(1), m.group(2)
        if prefix and any(c.isalpha() for c in prefix):
            val = _normalize_passport_alpha(prefix, digits)
            if card_number and val == card_number:
                return None
            return val
    compact = re.sub(r'[\s\W]+', '', text.upper())
    for m in PASSPORT_EMBEDDED_RE.finditer(compact):
        prefix, digits = m.group(1), m.group(2)
        if not any(c.isalpha() for c in prefix):
            continue
        val = _normalize_passport_alpha(prefix, digits)
        if card_number and (val == card_number or digits == card_number):
            continue
        if len(digits) >= 6:
            return val
    return None


def _parse_passport_text(text, card_number=None):
    """Return passport value: №digits, or letter+digit code."""
    if not text or not str(text).strip():
        return None
    text = str(text).strip()
    num = _parse_passport_numeric(text, card_number=card_number)
    if num:
        return num
    return _parse_passport_alpha(text, card_number=card_number)


def _parse_card_number_text(text):
    """Return registration card digits or None."""
    text = re.sub(r'^[№NnNo.\s]+', '', text.strip())
    text = re.sub(r'[^0-9]', '', text)
    if PURE_DIGITS_RE.match(text):
        return text
    return None


def _row_blocks_right_of_label(label, blocks, y_tol):
    lx, ly = _center(label['bbox'])
    label_right = max(p[0] for p in label['bbox'])
    row = []
    for b in blocks:
        if b is label:
            continue
        cy = _center(b['bbox'])[1]
        cx = _center(b['bbox'])[0]
        if abs(cy - ly) > y_tol:
            continue
        if cx < label_right - 40:
            continue
        row.append((cx, b))
    row.sort(key=lambda x: x[0])
    return [b for _, b in row]


def _find_label_block(blocks, keywords):
    for b in blocks:
        t = b['text'].lower()
        for kw in keywords:
            if kw in t or SequenceMatcher(None, t, kw).ratio() >= 0.72:
                return b
    return None


def _digit_blocks(blocks):
    """All OCR boxes that are (mostly) a card/passport digit string."""
    out = []
    for b in blocks:
        val = _parse_card_number_text(b['text'])
        if val:
            out.append((b, val))
    return out


def _score_registration_digit(val, b, blocks, passport_val):
    if passport_val and val == passport_val:
        return -1000.0
    yr, xr = _rel_y(b['bbox'], blocks), _rel_x(b['bbox'], blocks)
    score = b['confidence'] * 25
    # Internal passport numbers often start with 00 — not registration
    if val.startswith('00') and len(val) == 7:
        score -= 80
    if len(val) == 7 and not val.startswith('00'):
        score += 25
    if 6 <= len(val) <= 10:
        score += 8
    if yr <= 40:
        score += 12
    if xr >= 40:
        score += 8
    return score


def _score_passport_digit(val, b, blocks, on_passport_row=False):
    yr, xr = _rel_y(b['bbox'], blocks), _rel_x(b['bbox'], blocks)
    score = b['confidence'] * 25
    raw = b['text'].strip()
    if re.match(r'^[№NnNo]', raw):
        score += 50
    if on_passport_row:
        score += 70
    if val.startswith('00') and len(val) == 7:
        score += 35
    if yr <= 35 and xr >= 45:
        score += 15
    return score


def _passport_from_label_row(blocks, h, card_number=None):
    label = _find_label_block(blocks, PASSPORT_LABEL_KEYWORDS)
    if not label:
        return None
    y_tol = max(h * 0.04, 55)
    lx, ly = _center(label['bbox'])
    row_blocks = _row_blocks_right_of_label(label, blocks, y_tol)
    if row_blocks:
        merged = ' '.join(b['text'] for b in row_blocks)
        val = _parse_passport_text(merged, card_number=card_number)
        if val:
            conf = sum(b['confidence'] for b in row_blocks) / len(row_blocks)
            logger.info('ID extract: passport from шиноснома row: %r → %s', merged[:60], val)
            return (val, max(conf, 0.7))

    for b in blocks:
        text = b['text'].strip()
        cx, cy = _center(b['bbox'])
        if cx < lx - 30 or abs(cy - ly) > y_tol:
            continue
        val = _parse_passport_text(text, card_number=card_number)
        if val:
            return (val, max(b['confidence'], 0.65))
    return None


def _registration_from_label_row(blocks, h, passport_val=None):
    label = _find_label_block(blocks, REGISTRATION_LABEL_KEYWORDS)
    if not label:
        return None
    y_tol = max(h * 0.04, 55)
    row_blocks = _row_blocks_right_of_label(label, blocks, y_tol)
    if row_blocks:
        merged = ' '.join(b['text'] for b in row_blocks)
        val = _parse_card_number_text(merged)
        if val and val != passport_val:
            conf = sum(b['confidence'] for b in row_blocks) / len(row_blocks)
            logger.info('ID extract: registration from label row: %r → %s', merged[:60], val)
            return (val, max(conf, 0.7))
    return None


def extract_passport_number(blocks, w, h):
    """
    Field 2: № + digits (№0011801) or letter+digit, preferably «шиносномаи» row.
    """
    reg_guess = _registration_from_label_row(blocks, h)
    card_val = reg_guess[0] if reg_guess else None

    hit = _passport_from_label_row(blocks, h, card_number=card_val)
    if hit:
        return hit

    cands = []
    label = _find_label_block(blocks, PASSPORT_LABEL_KEYWORDS)
    on_row_vals = set()
    y_tol = max(h * 0.04, 55)
    if label:
        lx, ly = _center(label['bbox'])
        for b in blocks:
            cx, cy = _center(b['bbox'])
            if abs(cy - ly) <= y_tol and cx >= lx - 30:
                val = _parse_passport_text(b['text'], card_number=card_val)
                if val:
                    on_row_vals.add(
                        re.sub(r'[^0-9]', '', val) if val.startswith('№') else val
                    )
                    cands.append((_score_passport_digit(
                        re.sub(r'[^0-9]', '', val) if val.startswith('№') else val,
                        b, blocks, on_passport_row=True,
                    ) + 40, val, b['confidence']))

    for b, val in _digit_blocks(blocks):
        if card_val and val == card_val:
            continue
        formatted = format_passport_number(val)
        if not formatted:
            continue
        # Top-right 0011801 without label: still passport if 00-prefix
        if val not in on_row_vals and not (val.startswith('00') and len(val) == 7):
            continue
        score = _score_passport_digit(val, b, blocks)
        cands.append((score, formatted, b['confidence']))

    for b in blocks:
        val = _parse_passport_alpha(b['text'], card_number=card_val)
        if not val:
            continue
        yr = _rel_y(b['bbox'], blocks)
        if yr < 12 or yr > 68:
            continue
        cands.append((25 + b['confidence'] * 20, val, b['confidence']))

    if not cands:
        logger.warning('ID extract: passport_number not found')
        return None
    cands.sort(reverse=True)
    best = cands[0]
    logger.info('ID extract: passport_number=%s (score=%.0f)', best[1], best[0])
    return (best[1], max(best[2], 0.65))


def extract_registration_card_number(blocks, w, h):
    """
    Field 1: pure digits (e.g. 1233509), not the passport №0011801 / 0011801.
    """
    pass_guess = _passport_from_label_row(blocks, h)
    passport_val = None
    if pass_guess:
        passport_val = re.sub(r'[^0-9]', '', pass_guess[0])

    hit = _registration_from_label_row(blocks, h, passport_val=passport_val)
    if hit:
        return hit

    cands = []
    for b, val in _digit_blocks(blocks):
        if passport_val and val == passport_val:
            continue
        if val.startswith('00') and len(val) == 7 and not passport_val:
            continue
        score = _score_registration_digit(val, b, blocks, passport_val)
        if score < 0:
            continue
        cands.append((score, val, b['confidence']))

    if not cands:
        return None
    cands.sort(reverse=True)
    best = cands[0]
    logger.info('ID extract: registration_card_number=%s (score=%.0f)', best[1], best[0])
    return (best[1], max(best[2], 0.65))


def reconcile_id_fields(field_value_map):
    """
    Field 1 = registration digits; Field 2 = passport (№digits or alpha code).
    Fixes swaps when OCR put 0011801 into registration.
    """
    v1, c1 = field_value_map.get(1, ('', 0.0))
    v2, c2 = field_value_map.get(2, ('', 0.0))
    v1, v2 = (v1 or '').strip(), (v2 or '').strip()

    card1 = _parse_card_number_text(v1)
    card2 = _parse_card_number_text(v2)
    pass1 = _parse_passport_text(v1)
    pass2 = _parse_passport_text(v2)

    card = None
    passport = None
    conf_card = conf_pass = 0.0

    if card1 and pass2:
        card, passport, conf_card, conf_pass = card1, pass2, c1, c2
    elif card2 and pass1:
        logger.warning('ID reconcile: swapped slots corrected (%s <-> %s)', v1, v2)
        card, passport, conf_card, conf_pass = card2, pass1, c2, c1
    elif card1 and not pass2 and _parse_passport_numeric(v1):
        # Registration slot holds passport-shaped № / 00xxxx
        passport = _parse_passport_text(v1)
        conf_pass = c1
        card = card2
        conf_card = c2
    elif card2 and not pass1 and _parse_passport_numeric(v2):
        passport = _parse_passport_text(v2)
        conf_pass = c2
        card = card1
        conf_card = c1
    elif pass1 and not card2:
        passport, conf_pass = pass1, c1
        card, conf_card = card2, c2
    elif pass2 and not card1:
        passport, conf_pass = pass2, c2
        card, conf_card = card1, c1
    elif card1:
        card, conf_card = card1, c1
    elif card2:
        card, conf_card = card2, c2

    # Lone 0011801 in field 1 with empty field 2 → passport
    if card and not passport:
        maybe_pass = _parse_passport_text(card)
        if maybe_pass and maybe_pass != card and card.startswith('00'):
            passport = maybe_pass
            conf_pass = max(conf_card, conf_pass)
            card = card2 if card2 and card2 != re.sub(r'[^0-9]', '', passport) else None
            conf_card = c2 if card else 0.0

    if card:
        field_value_map[1] = (card, conf_card or max(c1, c2))
    elif v1 and (passport or _parse_passport_numeric(v1)):
        field_value_map[1] = ('', 0.0)
    if passport:
        field_value_map[2] = (passport, conf_pass or max(c1, c2))

    return field_value_map