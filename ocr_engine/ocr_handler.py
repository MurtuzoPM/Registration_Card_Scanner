import re
import logging

logger = logging.getLogger(__name__)


class OCRHandler:
    """Handles OCR text extraction using EasyOCR with Cyrillic/Tajik support.

    Uses multi-pass scanning with different parameter sets to maximize
    text extraction quality for Tajikistan registration cards.
    """

    def __init__(self, languages=None):
        self.languages = languages or ['ru', 'en']
        self.reader = None
        self._init_reader()

    def _init_reader(self):
        """Initialize the EasyOCR reader."""
        try:
            import easyocr
            logger.info('Initializing EasyOCR reader with languages: %s',
                        self.languages)
            self.reader = easyocr.Reader(self.languages, gpu=False)
            logger.info('EasyOCR reader initialized successfully')
        except ImportError:
            logger.error('easyocr package is not installed')
            raise
        except Exception as e:
            logger.error('Failed to initialize EasyOCR reader: %s', e)
            raise

    def extract_text(self, img_np):
        """
        Extract text from a NumPy image array using multi-pass OCR.
        Returns a list of dicts with keys: text, confidence, bbox.
        """
        if self.reader is None:
            raise RuntimeError('OCR reader not initialized')

        logger.info('Running multi-pass OCR on image...')

        # Pass 1: Fine-grained — individual text blocks, lower thresholds
        results_pass1 = self._run_ocr_pass(
            img_np,
            detail=1,
            paragraph=False,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.4,
            canvas_size=2560,
            mag_ratio=1.5,
            width_ths=0.5,
        )

        # Pass 2: High-res — better for small or faint text
        results_pass2 = self._run_ocr_pass(
            img_np,
            detail=1,
            paragraph=False,
            contrast_ths=0.05,
            adjust_contrast=0.3,
            text_threshold=0.4,
            low_text=0.2,
            link_threshold=0.3,
            canvas_size=3200,
            mag_ratio=2.5,
            width_ths=0.5,
        )

        # Pass 3: Wide merge — groups nearby text for multi-word values
        results_pass3 = self._run_ocr_pass(
            img_np,
            detail=1,
            paragraph=False,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.6,
            canvas_size=2560,
            mag_ratio=1.5,
            width_ths=0.9,
        )

        # Normalize all results to (bbox, text, confidence) tuples
        results_pass1 = self._normalize_results(results_pass1)
        results_pass2 = self._normalize_results(results_pass2)
        results_pass3 = self._normalize_results(results_pass3)

        # Merge results, preferring higher confidence
        all_results = self._merge_pass_results(
            results_pass1, results_pass2, results_pass3
        )

        # Filter and clean up
        extracted = []
        seen_texts = set()
        for bbox, text, confidence in all_results:
            text = self._clean_text(text)
            if not text or len(text) < 1:
                continue
            if confidence < 0.05:
                continue

            # Deduplicate near-identical texts
            text_key = text.lower().strip()
            if text_key in seen_texts:
                # Keep the higher confidence version
                for existing in extracted:
                    if existing['text'].lower().strip() == text_key:
                        if confidence > existing['confidence']:
                            existing['confidence'] = round(confidence, 3)
                            existing['bbox'] = [[int(p[0]), int(p[1])] for p in bbox]
                        break
                continue
            seen_texts.add(text_key)

            extracted.append({
                'text': text,
                'confidence': round(confidence, 3),
                'bbox': [[int(p[0]), int(p[1])] for p in bbox],
            })

        logger.info('OCR completed: %d text blocks extracted', len(extracted))
        return extracted

    def _normalize_results(self, results):
        """Ensure all OCR results are (bbox, text, confidence) 3-tuples.

        EasyOCR can return either:
          - (bbox, text, confidence) when detail=1
          - (bbox, text) when paragraph=True or detail=0
        This normalizes them all to 3-tuples.
        """
        normalized = []
        for item in results:
            if len(item) == 2:
                bbox, text = item
                confidence = 0.5
                normalized.append((bbox, text, confidence))
            elif len(item) == 3:
                normalized.append(item)
            elif len(item) >= 4:
                bbox, text, confidence = item[0], item[1], item[2]
                normalized.append((bbox, text, confidence))
            else:
                logger.warning('Unexpected OCR result format: %s', type(item))
        return normalized

    def _run_ocr_pass(self, img_np, **kwargs):
        """Run a single OCR pass with specified parameters."""
        try:
            results = self.reader.readtext(img_np, **kwargs)
            return results
        except Exception as e:
            logger.warning('OCR pass failed: %s', e)
            return []

    def _merge_pass_results(self, *result_sets):
        """Merge multiple OCR pass results, keeping the best text for each region."""
        all_results = []
        for results in result_sets:
            all_results.extend(results)

        if not all_results:
            return []

        merged = []
        used_indices = set()

        for i, (bbox_i, text_i, conf_i) in enumerate(all_results):
            if i in used_indices:
                continue

            best_idx = i
            best_conf = conf_i
            best_bbox = bbox_i
            best_text = text_i

            for j in range(i + 1, len(all_results)):
                if j in used_indices:
                    continue
                bbox_j, text_j, conf_j = all_results[j]

                if self._texts_overlap(text_i, text_j):
                    # Prefer longer text (more complete) or higher confidence
                    len_i = len(best_text.replace(' ', ''))
                    len_j = len(text_j.replace(' ', ''))
                    is_better = False
                    if len_j > len_i and len_j >= len_i * 1.3:
                        is_better = True
                    elif conf_j > best_conf and abs(len_j - len_i) / max(len_i, 1) < 0.3:
                        is_better = True
                    if is_better:
                        best_idx = j
                        best_conf = conf_j
                        best_bbox = bbox_j
                        best_text = text_j
                    used_indices.add(j)

            merged.append((best_bbox, all_results[best_idx][1], best_conf))
            used_indices.add(i)

        return merged

    def _texts_overlap(self, text1, text2):
        """Check if two text strings are similar enough to be duplicates."""
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        if t1 == t2:
            return True

        if t1 in t2 or t2 in t1:
            return True

        if len(t1) > 3 and len(t2) > 3:
            trigrams1 = set(t1[i:i+3] for i in range(len(t1) - 2))
            trigrams2 = set(t2[i:i+3] for i in range(len(t2) - 2))
            if trigrams1 and trigrams2:
                overlap = len(trigrams1 & trigrams2)
                union = len(trigrams1 | trigrams2)
                if union > 0 and overlap / union > 0.6:
                    return True

        return False

    def _clean_text(self, text):
        """Clean up OCR text artifacts common in registration cards."""
        if not text:
            return ''

        text = text.strip()

        # Fix common Unicode artifacts
        replacements = {
            '\u00a9': '\u0441',    # © → Cyrillic с
            '\u00ae': '\u0440',    # ® → Cyrillic р
            '\u2122': '\u0442',    # ™ → Cyrillic т
            '\u2014': '-',         # em dash
            '\u2013': '-',         # en dash
            '\u200b': '',          # zero-width space
            '\xa0': ' ',           # non-breaking space
            '\u20ac': 'E',         # € → E
            '\u0456': '\u0438',    # і → Cyrillic и
            '\u0457': '\u0451',    # ї → ё
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove leading/trailing non-alphanumeric tokens except for codes
        text = re.sub(r'^[\'"(*\s]+', '', text)
        text = re.sub(r'[\'")*\s]+$', '', text)

        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def extract_text_with_positions(self, img_np):
        """
        Extract text with detailed position information.
        Returns sorted list (top-to-bottom, left-to-right).
        """
        results = self.extract_text(img_np)
        results.sort(key=lambda x: (x['bbox'][0][1], x['bbox'][0][0]))
        return results
