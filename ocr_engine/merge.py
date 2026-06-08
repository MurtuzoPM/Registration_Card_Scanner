"""Merge OCR and parsed field results from multiple preprocessing passes."""


def merge_ocr_blocks(block_lists):
    """
    Combine OCR results from several image variants (same dimensions).
    Keeps the highest-confidence detection per exact text; for near-duplicate
    text at overlapping positions, keeps the longer / higher-confidence one.
    """
    if not block_lists:
        return []

    merged = []

    def _iou(a, b):
        ax1 = min(p[0] for p in a)
        ay1 = min(p[1] for p in a)
        ax2 = max(p[0] for p in a)
        ay2 = max(p[1] for p in a)
        bx1 = min(p[0] for p in b)
        by1 = min(p[1] for p in b)
        bx2 = max(p[0] for p in b)
        by2 = max(p[1] for p in b)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0

    for blocks in block_lists:
        for block in blocks:
            text = block.get('text', '').strip()
            if not text:
                continue
            conf = float(block.get('confidence', 0))
            bbox = block.get('bbox')
            if not bbox:
                continue

            replaced = False
            key_lower = text.lower()
            for i, existing in enumerate(merged):
                ex_text = existing['text']
                ex_key = ex_text.lower()
                overlap = (
                    key_lower == ex_key
                    or key_lower in ex_key
                    or ex_key in key_lower
                    or _iou(bbox, existing['bbox']) >= 0.35
                )
                if not overlap:
                    continue
                # Prefer longer text when confidence is similar; else higher conf
                pick_new = (
                    conf > existing['confidence'] + 0.08
                    or (conf >= existing['confidence'] - 0.05 and len(text) > len(ex_text))
                )
                if pick_new:
                    merged[i] = {
                        'text': text,
                        'confidence': conf,
                        'bbox': bbox,
                    }
                replaced = True
                break
            if not replaced:
                merged.append({
                    'text': text,
                    'confidence': conf,
                    'bbox': bbox,
                })

    merged.sort(key=lambda x: (x['bbox'][0][1], x['bbox'][0][0]))
    return merged


def merge_parsed_fields(field_dicts):
    """Pick the best value per field key across multiple parse passes."""
    best = {}
    for parsed in field_dicts:
        for key, data in parsed.items():
            if not data.get('value'):
                continue
            if key not in best or data['confidence'] > best[key]['confidence']:
                best[key] = data
    return best