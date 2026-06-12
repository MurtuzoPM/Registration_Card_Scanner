"""PaddleOCR backend.

PaddleOCR's PP-OCR models are considerably stronger than EasyOCR on small,
dense, multi-script print, and the Cyrillic recognizer covers the Russian /
Tajik alphabet. It runs on CPU (slower) or GPU.

This backend exposes the same contract as :class:`ocr_engine.ocr_handler.OCRHandler`
(``extract_text`` / ``extract_text_with_positions`` returning a list of
``{text, confidence, bbox}`` dicts), so it drops straight into the existing
TextParser / ROI / post-processing pipeline.

The package's Python API changed between the 2.x (``ocr.ocr(img)``) and 3.x
(``ocr.predict(img)``) lines, so the result parsing here is deliberately
defensive and supports both shapes.
"""

import logging

from config import Config
from .tajik_normalize import normalize_ocr_results

logger = logging.getLogger(__name__)


class PaddleOCRHandler:
    def __init__(self, languages=None, fast_mode=False, use_gpu=None):
        self.languages = languages or ['ru', 'en']
        self.fast_mode = fast_mode
        self.use_gpu = Config.OCR_USE_GPU if use_gpu is None else use_gpu
        # PaddleOCR uses a single recognizer per language family; the Cyrillic
        # model ("ru") covers Russian/Tajik letters. Latin digits/letters are
        # recognised by the same detector, so "ru" is the right primary lang.
        self.lang = self._resolve_lang(self.languages)
        self._reader = None
        self._predict_fn = None
        self._init_engine()

    @staticmethod
    def _resolve_lang(languages):
        langs = [l.lower() for l in (languages or [])]
        if any(l in ('ru', 'tg', 'tgk', 'cyrillic', 'rus') for l in langs):
            return 'ru'
        if 'en' in langs:
            return 'en'
        return 'ru'

    def _auto_detect_gpu(self):
        try:
            import paddle
            return bool(paddle.device.cuda.device_count() > 0)
        except Exception:
            try:
                import torch
                return bool(torch.cuda.is_available())
            except Exception:
                return False

    def _resolve_gpu(self):
        if self.use_gpu is None:
            return self._auto_detect_gpu()
        return bool(self.use_gpu)

    def _init_engine(self):
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                'PaddleOCR is not installed. Install it with '
                '`pip install -r requirements-paddle.txt`'
            ) from exc

        gpu = self._resolve_gpu()
        # Try the most compatible constructor first, then fall back across the
        # 2.x / 3.x keyword differences without crashing.
        last_err = None
        for kwargs in (
            {'use_angle_cls': True, 'lang': self.lang, 'use_gpu': gpu, 'show_log': False},
            {'use_textline_orientation': True, 'lang': self.lang},
            {'lang': self.lang},
        ):
            try:
                self._reader = PaddleOCR(**kwargs)
                break
            except Exception as exc:  # pragma: no cover - version probing
                last_err = exc
                continue
        if self._reader is None:
            raise RuntimeError(f'Could not initialise PaddleOCR: {last_err}')

        # Pick the inference method available in this version.
        if hasattr(self._reader, 'predict'):
            self._predict_fn = self._reader.predict
        else:
            self._predict_fn = self._reader.ocr
        logger.info('PaddleOCR initialised (lang=%s, gpu=%s)', self.lang, gpu)

    # ------------------------------------------------------------------
    # Result parsing (supports 2.x list shape and 3.x dict shape)
    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_to_ints(box):
        return [[int(round(p[0])), int(round(p[1]))] for p in box]

    @classmethod
    def parse_result(cls, result):
        """Convert a PaddleOCR result into ``[(bbox, text, confidence), ...]``.

        Handles:
          * 2.x: ``[[ [box,(text,score)], ... ]]`` or ``[ [box,(text,score)], ... ]``
          * 3.x: ``[{'rec_texts': [...], 'rec_scores': [...], 'rec_polys'/'dt_polys': [...]}]``
        """
        out = []
        if not result:
            return out

        # 3.x dict shape -------------------------------------------------
        first = result[0] if isinstance(result, (list, tuple)) and result else result
        if isinstance(first, dict):
            data = first
            texts = data.get('rec_texts') or data.get('rec_text') or []
            scores = data.get('rec_scores') or data.get('rec_score') or []
            polys = data.get('rec_polys') or data.get('dt_polys') or data.get('rec_boxes') or []
            for i, text in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 0.5
                box = polys[i] if i < len(polys) else None
                if box is None:
                    continue
                # rec_boxes may be [x1,y1,x2,y2]; polys are 4-point
                if len(box) == 4 and not hasattr(box[0], '__len__'):
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                else:
                    bbox = cls._bbox_to_ints(box)
                out.append((bbox, str(text), score))
            return out

        # 2.x list shape -------------------------------------------------
        dets = result[0] if (isinstance(result[0], list) and len(result) == 1) else result
        for det in dets or []:
            try:
                box = det[0]
                text, score = det[1][0], float(det[1][1])
            except (IndexError, TypeError, ValueError):
                continue
            out.append((cls._bbox_to_ints(box), str(text), score))
        return out

    # ------------------------------------------------------------------
    # Public API (mirrors OCRHandler)
    # ------------------------------------------------------------------
    def extract_text(self, img_np):
        try:
            result = self._predict_fn(img_np)
        except TypeError:
            # 2.x accepts a `cls` kwarg; 3.x predict does not.
            result = self._reader.ocr(img_np, cls=True)
        except Exception as exc:
            logger.warning('PaddleOCR run failed: %s', exc)
            return []

        parsed = self.parse_result(result)
        extracted = []
        seen = {}
        for bbox, text, confidence in parsed:
            text = (text or '').strip()
            if not text or confidence < 0.05:
                continue
            entry = {
                'text': text,
                'confidence': round(float(confidence), 3),
                'bbox': bbox,
            }
            key = text.lower()
            if key in seen:
                if entry['confidence'] > seen[key]['confidence']:
                    seen[key].update(entry)
                continue
            seen[key] = entry
            extracted.append(entry)
        logger.info('PaddleOCR: %d unique text blocks', len(extracted))
        return normalize_ocr_results(extracted)

    def extract_text_with_positions(self, img_np):
        results = self.extract_text(img_np)
        results.sort(key=lambda x: (x['bbox'][0][1], x['bbox'][0][0]))
        return results
