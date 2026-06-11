#!/usr/bin/env python3
"""Run one image through the same extraction pipeline and save debug output.

Usage:
  python tools/debug_extract.py training/images/card_001.png --out training/reports/card_001_debug.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from ocr_engine import (  # noqa: E402
    OCRHandler,
    TextParser,
    extract_roi_fields,
    merge_ocr_blocks,
    merge_parsed_fields,
    preprocess_image,
    validate_image,
    postprocess_fields,
)


def _is_valid_field(data):
    return isinstance(data, dict) and "value" in data


def parse_image(image_path: Path, fast_mode: bool):
    ok, msg = validate_image(str(image_path))
    if not ok:
        raise RuntimeError(f"Invalid image {image_path}: {msg}")

    ocr_handler = OCRHandler(languages=Config.OCR_LANGUAGES, fast_mode=fast_mode)
    text_parser = TextParser()

    processed_images = preprocess_image(
        str(image_path),
        enable_preprocessing=Config.ENABLE_IMAGE_PREPROCESSING,
        fast_mode=fast_mode,
    )

    per_pass_ocr = []
    per_pass_parsed = []
    for img in processed_images:
        ocr_results = ocr_handler.extract_text_with_positions(img)
        per_pass_ocr.append(ocr_results)
        parsed = text_parser.parse(ocr_results) or {}
        per_pass_parsed.append({k: v for k, v in parsed.items() if _is_valid_field(v)})

    best_fields = {
        k: v for k, v in (merge_parsed_fields(per_pass_parsed) or {}).items()
        if _is_valid_field(v)
    }

    merged_ocr = merge_ocr_blocks(per_pass_ocr)
    parsed_merged = text_parser.parse(merged_ocr) or {}
    for key, data in parsed_merged.items():
        if _is_valid_field(data) and data.get("value"):
            old = best_fields.get(key)
            if not _is_valid_field(old) or float(data.get("confidence", 0)) > float(old.get("confidence", 0)):
                best_fields[key] = data

    roi_fields, roi_ocr, roi_debug = extract_roi_fields(processed_images, ocr_handler, merged_ocr)
    for key, data in (roi_fields or {}).items():
        if _is_valid_field(data) and data.get("value"):
            old = best_fields.get(key)
            if not _is_valid_field(old) or float(data.get("confidence", 0)) >= float(old.get("confidence", 0)) - 0.12:
                best_fields[key] = data

    all_ocr = merge_ocr_blocks([merged_ocr, roi_ocr])

    parsed_data = {}
    for field_num, field_info in Config.FIELDS.items():
        key = field_info["key"]
        data = best_fields.get(key)
        if _is_valid_field(data):
            parsed_data[key] = data
        else:
            parsed_data[key] = {
                "value": "",
                "confidence": 0.0,
                "label": field_info["label"],
                "field_number": field_num,
            }

    parsed_data = postprocess_fields(parsed_data, all_ocr)

    fields = {}
    for key, data in parsed_data.items():
        fields[key] = {
            "value": data.get("value", ""),
            "confidence": data.get("confidence", 0.0),
            "label": data.get("label", key),
            "field_number": data.get("field_number"),
            "source": data.get("source"),
        }

    return {
        "image": str(image_path),
        "fast_mode": fast_mode,
        "fields": fields,
        "roi_debug": roi_debug,
        "raw_ocr": [
            {
                "text": r.get("text", ""),
                "confidence": r.get("confidence", 0.0),
                "bbox": r.get("bbox"),
                "roi_field": r.get("roi_field"),
            }
            for r in all_ocr
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Debug OCR extraction for one registration-card image.")
    parser.add_argument("image", help="Image path, e.g. training/images/card_001.png")
    parser.add_argument("--out", default="training/reports/debug_extract.json", help="Output JSON file")
    parser.add_argument("--slow", action="store_true", help="Use slow/high-recall mode")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = (ROOT / image_path).resolve()

    result = parse_image(image_path, fast_mode=not args.slow)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result["fields"], ensure_ascii=False, indent=2))
    print(f"Debug report written to: {out_path}")


if __name__ == "__main__":
    main()
