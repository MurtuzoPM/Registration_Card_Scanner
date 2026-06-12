#!/usr/bin/env python3
"""Evaluate registration-card OCR against a local labelled dataset.

This is a local training/evaluation helper. It does not upload images anywhere.
Keep real card images and labels outside Git or in ignored paths.

Dataset JSON format:
{
  "samples": [
    {
      "image": "training/images/card_001.png",
      "fields": {
        "registration_card_number": "1234567",
        "passport_number": "P06476028"
      }
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from ocr_engine import (  # noqa: E402
    OCRHandler,
    TextParser,
    create_ocr_handler,
    extract_roi_fields,
    merge_ocr_blocks,
    merge_parsed_fields,
    preprocess_image,
    validate_image,
    postprocess_fields,
)



def _is_valid_field(data: Any) -> bool:
    return isinstance(data, dict) and "value" in data


def _norm_value(value: Any) -> str:
    value = "" if value is None else str(value)
    value = value.strip()
    value = value.replace("ё", "е").replace("Ё", "Е")
    value = value.replace("ӯ", "у").replace("Ӯ", "У")
    value = value.replace("ӣ", "и").replace("Ӣ", "И")
    value = value.replace("ҳ", "х").replace("Ҳ", "Х")
    value = value.replace("ҷ", "ч").replace("Ҷ", "Ч")
    value = value.replace("қ", "к").replace("Қ", "К")
    value = value.replace("ғ", "г").replace("Ғ", "Г")
    value = re.sub(r"\s+", " ", value)
    return value


def _score(expected: str, actual: str) -> float:
    expected = _norm_value(expected)
    actual = _norm_value(actual)
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    if expected == actual:
        return 1.0
    exp_compact = re.sub(r"[\s./\-№.,:;()]", "", expected).upper()
    act_compact = re.sub(r"[\s./\-№.,:;()]", "", actual).upper()
    if exp_compact and exp_compact == act_compact:
        return 1.0
    if exp_compact and (exp_compact in act_compact or act_compact in exp_compact):
        shorter = min(len(exp_compact), len(act_compact))
        longer = max(len(exp_compact), len(act_compact))
        if shorter >= 4:
            return shorter / longer
    return SequenceMatcher(None, expected.lower(), actual.lower()).ratio()


def _prefer(existing: Dict[str, Any] | None, candidate: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not _is_valid_field(candidate) or not candidate.get("value"):
        return existing if _is_valid_field(existing) else None
    if not _is_valid_field(existing) or not existing.get("value"):
        return candidate
    if float(candidate.get("confidence", 0)) >= float(existing.get("confidence", 0)) - 0.12:
        return candidate
    return existing


def _parse_image(path: Path, ocr_handler: OCRHandler, text_parser: TextParser, fast_mode: bool | None = None) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    ok, msg = validate_image(str(path))
    if not ok:
        raise RuntimeError(f"Invalid image {path}: {msg}")

    if fast_mode is None:
        fast_mode = Config.OCR_FAST_MODE

    processed_images = preprocess_image(
        str(path),
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
            best_fields[key] = _prefer(best_fields.get(key), data)

    roi_fields, roi_ocr, roi_debug = extract_roi_fields(processed_images, ocr_handler, merged_ocr)
    for key, data in (roi_fields or {}).items():
        if _is_valid_field(data) and data.get("value"):
            best_fields[key] = _prefer(best_fields.get(key), data)

    merged_all_ocr = merge_ocr_blocks([merged_ocr, roi_ocr])
    fields = {}
    for field_num, field_info in Config.FIELDS.items():
        key = field_info["key"]
        data = best_fields.get(key)
        if _is_valid_field(data):
            fields[key] = data
        else:
            fields[key] = {
                "value": "",
                "confidence": 0.0,
                "label": field_info["label"],
                "field_number": field_num,
            }

    fields = postprocess_fields(fields, merged_all_ocr)
    return fields, {"raw_ocr": merged_all_ocr, "roi_debug": roi_debug}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate OCR extraction against labelled local images.")
    parser.add_argument("--labels", default="training/labels.local.json", help="Path to local labels JSON")
    parser.add_argument("--fast", action="store_true", help="Force fast OCR mode")
    parser.add_argument("--slow", action="store_true", help="Force slow/high-recall OCR mode")
    parser.add_argument("--report", default="training/reports/evaluation.json", help="Where to write detailed report")
    args = parser.parse_args()

    labels_path = (ROOT / args.labels).resolve()
    with labels_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    fast_mode = None
    if args.fast:
        fast_mode = True
    if args.slow:
        fast_mode = False

    # Initialize OCR only once for the whole dataset. This is much faster on CPU.
    ocr_handler = create_ocr_handler(languages=Config.OCR_LANGUAGES, fast_mode=fast_mode if fast_mode is not None else Config.OCR_FAST_MODE)
    text_parser = TextParser()

    report = {"samples": [], "summary": {}}
    field_scores: Dict[str, list[float]] = {}

    for idx, sample in enumerate(dataset.get("samples", []), start=1):
        image_path = (labels_path.parent / sample["image"]).resolve() if not os.path.isabs(sample["image"]) else Path(sample["image"])
        if not image_path.exists():
            image_path = (ROOT / sample["image"]).resolve()
        print(f"[{idx}/{len(dataset.get('samples', []))}] Processing {sample['image']}...", flush=True)
        fields, debug = _parse_image(image_path, ocr_handler=ocr_handler, text_parser=text_parser, fast_mode=fast_mode)
        expected = sample.get("fields", {})
        actual = {k: v.get("value", "") for k, v in fields.items()}

        sample_result = {
            "image": sample["image"],
            "expected": expected,
            "actual": actual,
            "fields": {},
        }
        for key, exp in expected.items():
            act = actual.get(key, "")
            score = _score(exp, act)
            sample_result["fields"][key] = {"expected": exp, "actual": act, "score": round(score, 3)}
            field_scores.setdefault(key, []).append(score)
        report["samples"].append(sample_result)

    summary = {}
    for key, scores in field_scores.items():
        summary[key] = round(sum(scores) / len(scores), 3) if scores else 0.0
    summary["overall"] = round(sum(summary.values()) / len(summary), 3) if summary else 0.0
    report["summary"] = summary

    report_path = (ROOT / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Detailed report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
