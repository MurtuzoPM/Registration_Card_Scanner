import re
import logging
from config import Config
from .tajik_normalize import normalize_ocr_results

logger = logging.getLogger(__name__)


class OCRHandler:
    """Hybrid OCR: runs Tesseract AND EasyOCR, merges results.

    Rationale: on Tajik registration cards, Tesseract is strong on numerics
    (card numbers, dates) while EasyOCR is strong on Cyrillic words (names,
    place names). Running both and merging the de-duplicated results gives
    higher recall than either alone.

    Output format preserved: list of dicts with text, confidence, bbox.
    """

    TESS_LANGS = 'rus+tgk+eng'

    def __init__(self, languages=None, fast_mode=False, use_gpu=None):
        self.languages = languages or ['ru', 'en']
        self.fast_mode = fast_mode
        self.use_gpu = Config.OCR_USE_GPU if use_gpu is None else use_gpu
        self.easyocr_gpu = False
        self.tesseract = None
        self.reader = None
        self.engines = []
        self._init_engines()

    def _auto_detect_gpu(self):
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _resolve_easyocr_gpu(self):
        if self.use_gpu is None:
            return self._auto_detect_gpu()
        return bool(self.use_gpu)

    def _init_engines(self):
        # Tesseract
        try:
            import pytesseract
            available = pytesseract.get_languages(config='')
            if 'rus' not in available:
                raise RuntimeError("Tesseract 'rus' pack missing")
            self.tesseract = pytesseract
            if 'tgk' not in available:
                self.TESS_LANGS = 'rus+eng'
            self.engines.append('tesseract')
            logger.info('Tesseract initialized (langs=%s)', self.TESS_LANGS)
        except Exception as exc:
            logger.warning('Tesseract unavailable: %s', exc)

        # EasyOCR
        try:
            import easyocr
            use_gpu = self._resolve_easyocr_gpu()
            self.reader = easyocr.Reader(self.languages, gpu=use_gpu)
            self.easyocr_gpu = use_gpu
            self.engines.append('easyocr')
            logger.info('EasyOCR initialized (gpu=%s)', use_gpu)
        except Exception as exc:
            logger.warning('EasyOCR unavailable: %s', exc)

        if not self.engines:
            raise RuntimeError('No OCR engine available — install pytesseract or easyocr')
        logger.info('Hybrid OCR active with engines: %s', self.engines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_text(self, img_np):
        all_raw = []
        if 'tesseract' in self.engines:
            try:
                all_raw += [(*r, 'tesseract') for r in self._run_tesseract(img_np)]
            except Exception as exc:
                logger.warning('Tesseract run failed: %s', exc)
        if 'easyocr' in self.engines:
            try:
                all_raw += [(*r, 'easyocr') for r in self._run_easyocr(img_np)]
            except Exception as exc:
                logger.warning('EasyOCR run failed: %s', exc)

        merged = self._merge_cross_engine(all_raw)

        extracted = []
        seen_keys = {}
        for bbox, text, confidence, source in merged:
            text = self._clean_text(text)
            if not text or confidence < 0.05:
                continue
            key = text.lower().strip()
            entry = {
                'text': text,
                'confidence': round(float(confidence), 3),
                'bbox': [[int(p[0]), int(p[1])] for p in bbox],
            }
            if key in seen_keys:
                if entry['confidence'] > seen_keys[key]['confidence']:
                    seen_keys[key].update(entry)
                continue
            seen_keys[key] = entry
            extracted.append(entry)
        logger.info('Hybrid OCR: %d unique text blocks (from %d raw)',
                    len(extracted), len(all_raw))
        return normalize_ocr_results(extracted)

    def extract_text_with_positions(self, img_np):
        results = self.extract_text(img_np)
        results.sort(key=lambda x: (x['bbox'][0][1], x['bbox'][0][0]))
        return results

    # ------------------------------------------------------------------
    # Tesseract backend
    # ------------------------------------------------------------------
    def _run_tesseract(self, img_np):
        psms = [6, 11, 4]  # 4 = single column — helps dates/numbers on cards
        out = []
        for psm in psms:
            try:
                cfg = f'--oem 3 --psm {psm}'
                data = self.tesseract.image_to_data(
                    img_np, lang=self.TESS_LANGS, config=cfg,
                    output_type=self.tesseract.Output.DICT,
                )
            except Exception as exc:
                logger.warning('Tesseract PSM %d failed: %s', psm, exc)
                continue
            n = len(data.get('text', []))
            for i in range(n):
                text = (data['text'][i] or '').strip()
                if not text:
                    continue
                try:
                    conf = float(data['conf'][i])
                except (ValueError, TypeError):
                    conf = -1
                if conf < 25:
                    continue
                x, y = int(data['left'][i]), int(data['top'][i])
                w, h = int(data['width'][i]), int(data['height'][i])
                bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                out.append((bbox, text, conf / 100.0))

        # Supplemental digit-focused pass (local, no external API)
        try:
            cfg = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789./№-'
            data = self.tesseract.image_to_data(
                img_np, lang='eng', config=cfg,
                output_type=self.tesseract.Output.DICT,
            )
            n = len(data.get('text', []))
            for i in range(n):
                text = (data['text'][i] or '').strip()
                if not text or len(text) < 3:
                    continue
                try:
                    conf = float(data['conf'][i])
                except (ValueError, TypeError):
                    conf = -1
                if conf < 30:
                    continue
                x, y = int(data['left'][i]), int(data['top'][i])
                w, h = int(data['width'][i]), int(data['height'][i])
                bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                out.append((bbox, text, conf / 100.0))
        except Exception as exc:
            logger.debug('Tesseract digit pass skipped: %s', exc)

        return out

    # ------------------------------------------------------------------
    # EasyOCR backend
    # ------------------------------------------------------------------
    def _run_easyocr(self, img_np):
        p1 = self._easyocr_pass(img_np, canvas_size=2560, mag_ratio=1.5,
                                text_threshold=0.5, low_text=0.3,
                                link_threshold=0.4, width_ths=0.5)
        if self.fast_mode:
            return self._merge_overlapping(self._normalize(p1))
        p2 = self._easyocr_pass(img_np, canvas_size=3200, mag_ratio=2.5,
                                text_threshold=0.4, low_text=0.2,
                                link_threshold=0.3, width_ths=0.5)
        raw = self._normalize(p1) + self._normalize(p2)
        return self._merge_overlapping(raw)

    def _easyocr_pass(self, img_np, **kwargs):
        try:
            return self.reader.readtext(
                img_np, detail=1, paragraph=False,
                contrast_ths=0.1, adjust_contrast=0.5, **kwargs,
            )
        except Exception as e:
            logger.warning('EasyOCR pass failed: %s', e)
            return []

    def _normalize(self, results):
        out = []
        for item in results:
            if len(item) == 2:
                out.append((item[0], item[1], 0.5))
            elif len(item) >= 3:
                out.append((item[0], item[1], item[2]))
        return out

    # ------------------------------------------------------------------
    # Cross-engine merge: per-region take the higher-confidence detection,
    # but for purely-numeric text prefer Tesseract; for Cyrillic words
    # prefer EasyOCR.
    # ------------------------------------------------------------------
    def _bbox_iou(self, a, b):
        ax1 = min(p[0] for p in a); ay1 = min(p[1] for p in a)
        ax2 = max(p[0] for p in a); ay2 = max(p[1] for p in a)
        bx1 = min(p[0] for p in b); by1 = min(p[1] for p in b)
        bx2 = max(p[0] for p in b); by2 = max(p[1] for p in b)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0

    def _is_mostly_numeric(self, s):
        digits = sum(c.isdigit() for c in s)
        return len(s) > 0 and digits / len(s) >= 0.6

    def _is_cyrillic_word(self, s):
        cyr = sum(1 for c in s if '\u0400' <= c <= '\u04ff')
        return len(s) > 2 and cyr / max(len(s), 1) >= 0.5

    def _merge_cross_engine(self, all_raw):
        if not all_raw:
            return []
        merged = []
        used = set()
        for i, (bb_i, t_i, c_i, src_i) in enumerate(all_raw):
            if i in used:
                continue
            best = (bb_i, t_i, c_i, src_i)
            used.add(i)
            for j in range(i + 1, len(all_raw)):
                if j in used:
                    continue
                bb_j, t_j, c_j, src_j = all_raw[j]
                if self._bbox_iou(best[0], bb_j) < 0.3 and \
                        not self._texts_overlap(best[1], t_j):
                    continue
                used.add(j)
                # Decide which wins
                a_num = self._is_mostly_numeric(best[1])
                b_num = self._is_mostly_numeric(t_j)
                a_cyr = self._is_cyrillic_word(best[1])
                b_cyr = self._is_cyrillic_word(t_j)
                # Numeric: prefer Tesseract
                if a_num or b_num:
                    if best[3] == 'tesseract' and src_j != 'tesseract':
                        pass
                    elif src_j == 'tesseract' and best[3] != 'tesseract':
                        best = (bb_j, t_j, c_j, src_j)
                    elif c_j > best[2]:
                        best = (bb_j, t_j, c_j, src_j)
                # Cyrillic: prefer EasyOCR
                elif a_cyr or b_cyr:
                    rank = {'easyocr': 1, 'tesseract': 0}
                    if rank.get(src_j, 0) > rank.get(best[3], 0):
                        best = (bb_j, t_j, c_j, src_j)
                    elif rank.get(src_j, 0) == rank.get(best[3], 0) and c_j > best[2]:
                        best = (bb_j, t_j, c_j, src_j)
                else:
                    if c_j > best[2]:
                        best = (bb_j, t_j, c_j, src_j)
            merged.append(best)
        return merged

    def _merge_overlapping(self, all_results):
        if not all_results:
            return []
        merged = []
        used = set()
        for i, (bb_i, t_i, c_i) in enumerate(all_results):
            if i in used:
                continue
            best = (bb_i, t_i, c_i)
            used.add(i)
            for j in range(i + 1, len(all_results)):
                if j in used:
                    continue
                bb_j, t_j, c_j = all_results[j]
                if self._texts_overlap(best[1], t_j):
                    if (len(t_j) > len(best[1]) * 1.3) or \
                       (c_j > best[2] and abs(len(t_j) - len(best[1])) / max(len(best[1]), 1) < 0.3):
                        best = (bb_j, t_j, c_j)
                    used.add(j)
            merged.append(best)
        return merged

    def _texts_overlap(self, a, b):
        a = a.lower().strip(); b = b.lower().strip()
        if a == b or a in b or b in a:
            return True
        if len(a) > 3 and len(b) > 3:
            t1 = set(a[i:i+3] for i in range(len(a)-2))
            t2 = set(b[i:i+3] for i in range(len(b)-2))
            u = len(t1 | t2)
            return u > 0 and len(t1 & t2) / u > 0.6
        return False

    def _clean_text(self, text):
        if not text:
            return ''
        text = text.strip()
        replacements = {
            '\u00a9': '\u0441', '\u00ae': '\u0440', '\u2122': '\u0442',
            '\u2014': '-', '\u2013': '-', '\u200b': '', '\xa0': ' ',
            '\u20ac': 'E', '\u0456': '\u0438', '\u0457': '\u0451',
        }
        for o, n in replacements.items():
            text = text.replace(o, n)
        text = re.sub(r"^['\"(*\s]+", '', text)
        text = re.sub(r"['\")*\s]+$", '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
