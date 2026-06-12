"""Chandra vision-LLM field extractor (experimental, GPU recommended).

Chandra (https://huggingface.co/datalab-to/chandra) is a document OCR *model*
(image-text-to-text). Unlike PaddleOCR/EasyOCR it does not return per-word
boxes; it reads the whole page. The strongest way to use it for this fixed
set of fields is *end-to-end*: show it the card and ask for the 13 fields as
JSON, skipping the TextParser/ROI pipeline entirely.

This module keeps the heavy model import lazy. The prompt building and JSON
parsing are pure functions so they can be unit-tested without downloading the
model.

Usage:
    from ocr_engine.chandra_backend import ChandraFieldExtractor
    extractor = ChandraFieldExtractor()           # loads the model (slow)
    fields = extractor.extract_fields('card.png') # -> {key: {value, confidence, ...}}
"""

import json
import logging
import re

from config import Config

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'datalab-to/chandra'


def build_prompt():
    """Build an instruction listing the expected fields (Tajik label + key)."""
    lines = [
        'You are reading a Tajikistan foreigner registration card.',
        'Extract these fields and respond with ONLY a JSON object whose keys are',
        'the english keys below and whose values are the exact text on the card',
        '(empty string "" if a field is absent). Do not add commentary.',
        '',
    ]
    for num in sorted(Config.FIELDS):
        info = Config.FIELDS[num]
        lines.append(f'- {info["key"]}: {info["label"]} ({info["description"]})')
    lines.append('')
    lines.append('Return JSON only.')
    return '\n'.join(lines)


def parse_model_json(raw_text):
    """Extract the first JSON object from the model output and return a dict.

    Tolerant of code fences and surrounding prose.
    """
    if not raw_text:
        return {}
    text = raw_text.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S | re.I)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r'\{.*\}', text, re.S)
        if brace:
            text = brace.group(0)
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def map_to_fields(data, confidence=0.9):
    """Map a raw key->value dict to the app's field structure."""
    valid_keys = {info['key']: num for num, info in Config.FIELDS.items()}
    fields = {}
    for num, info in Config.FIELDS.items():
        key = info['key']
        value = data.get(key, '')
        if not isinstance(value, str):
            value = '' if value is None else str(value)
        value = value.strip()
        fields[key] = {
            'value': value,
            'confidence': round(confidence, 3) if value else 0.0,
            'label': info['label'],
            'field_number': num,
            'source': 'chandra',
        }
    return fields


class ChandraFieldExtractor:
    def __init__(self, model_name=DEFAULT_MODEL, device=None, max_new_tokens=2048):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                'transformers is required for the Chandra backend. Install with '
                '`pip install -r requirements-chandra.txt`'
            ) from exc
        kwargs = {'model': self.model_name}
        if self.device is not None:
            kwargs['device'] = self.device
        logger.info('Loading Chandra model %s (this can take a while)...', self.model_name)
        self._pipe = pipeline('image-text-to-text', **kwargs)

    def extract_fields(self, image_path):
        self._ensure_loaded()
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'url': str(image_path)},
                {'type': 'text', 'text': build_prompt()},
            ],
        }]
        result = self._pipe(text=messages, max_new_tokens=self.max_new_tokens)
        raw = self._extract_generated_text(result)
        data = parse_model_json(raw)
        if not data:
            logger.warning('Chandra returned no parseable JSON; raw head: %s', (raw or '')[:200])
        return map_to_fields(data)

    @staticmethod
    def _extract_generated_text(result):
        """Pull the assistant text out of a transformers pipeline result."""
        if isinstance(result, str):
            return result
        if isinstance(result, list) and result:
            item = result[0]
            if isinstance(item, dict):
                gen = item.get('generated_text')
                if isinstance(gen, str):
                    return gen
                if isinstance(gen, list) and gen:
                    last = gen[-1]
                    if isinstance(last, dict):
                        content = last.get('content')
                        if isinstance(content, str):
                            return content
                        if isinstance(content, list):
                            return ' '.join(
                                c.get('text', '') for c in content if isinstance(c, dict)
                            )
                    return str(last)
        return str(result)
