"""Factory for selecting an OCR engine.

The engine is chosen (in priority order) by:
  1. the explicit ``engine=`` argument, then
  2. the ``OCR_ENGINE`` environment variable / ``Config.OCR_ENGINE``.

Supported values:
  * ``hybrid`` / ``easyocr`` (default) - Tesseract + EasyOCR (CPU friendly).
  * ``paddleocr`` / ``paddle`` / ``ppocr`` - PaddleOCR PP-OCR (CPU or GPU).
  * ``chandra`` - reserved for the Chandra vision-LLM extractor, which is an
    end-to-end field extractor and is invoked separately (see
    ``ocr_engine.chandra_backend``); it is not a drop-in block engine.

All returned handlers expose ``extract_text_with_positions(img_np)``.
"""

import logging

from config import Config
from .ocr_handler import OCRHandler

logger = logging.getLogger(__name__)

_HYBRID = {'hybrid', 'easyocr', 'tesseract', 'default', ''}
_PADDLE = {'paddleocr', 'paddle', 'ppocr', 'pp-ocr', 'pp_ocr'}


def create_ocr_handler(engine=None, languages=None, fast_mode=False, use_gpu=None):
    name = (engine or getattr(Config, 'OCR_ENGINE', 'hybrid') or 'hybrid').strip().lower()

    if name in _PADDLE:
        from .paddle_backend import PaddleOCRHandler
        logger.info('Using PaddleOCR engine')
        return PaddleOCRHandler(languages=languages, fast_mode=fast_mode, use_gpu=use_gpu)

    if name == 'chandra':
        raise ValueError(
            "Chandra is an end-to-end vision-LLM extractor, not a block OCR "
            "engine. Use ocr_engine.chandra_backend.ChandraFieldExtractor "
            "(see tools/extract_chandra.py) instead of create_ocr_handler('chandra')."
        )

    if name not in _HYBRID:
        logger.warning("Unknown OCR_ENGINE '%s'; falling back to hybrid.", name)
    return OCRHandler(languages=languages, fast_mode=fast_mode, use_gpu=use_gpu)
