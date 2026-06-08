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
    extract_roi_fields,
    merge_ocr_blocks,
    merge_parsed_fields,
    preprocess_image,
    validate_image,
)

DATE_RE = re.compile(r'(?<!\d)(\d{1,2})[./,\-\s]+(\d{1,2})[./,\-\s]+(\d{2,4})(?!\d)')
DIGIT_FIX = str.maketrans({
    'O': '0', 'О': '0', 'o': '0', 'о': '0',
    'I': '1', 'l': '1', 'L': '1', 'S': '5',
    'З': '3', 'з': '3', 'B': '8',
})


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


def _field_template(key: str, value: str, confidence: float = 0.78, source: str = "eval_postprocess") -> Dict[str, Any]:
    for num, info in Config.FIELDS.items():
        if info["key"] == key:
            return {
                "value": value,
                "confidence": round(float(confidence), 3),
                "label": info["label"],
                "field_number": num,
                "source": source,
            }
    return {"value": value, "confidence": confidence, "label": key, "field_number": None, "source": source}


def _bbox_center(block: Dict[str, Any]) -> Tuple[float, float]:
    bbox = block.get("bbox") or [[0, 0], [0, 0]]
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _norm_date(text: str) -> str | None:
    m = DATE_RE.search((text or "").translate(DIGIT_FIX))
    if not m:
        return None
    d, mo, y = m.groups()
    d, mo = d.zfill(2), mo.zfill(2)
    if len(y) == 2:
        y = "20" + y if int(y) < 80 else "19" + y
    if 1 <= int(d) <= 31 and 1 <= int(mo) <= 12:
        return f"{d}.{mo}.{y}"
    return None


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _looks_like_date_or_year(value: str, dates: list[str]) -> bool:
    digs = _digits(value)
    if not digs:
        return False
    if any(y in digs for y in ("2024", "2025", "2026", "2027")):
        return True
    return any(digs in _digits(d) or _digits(d) in digs for d in dates if d)


def _postprocess_like_app(fields: Dict[str, Dict[str, Any]], ocr_results: list[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Apply the same type of final sanity checks that the Flask app applies."""
    raw = " ".join(str(r.get("text", "")) for r in ocr_results or [])
    raw_l = raw.lower()

    name = fields.get("name_and_surname", {}).get("value", "")
    if name:
        name = re.sub(r"\b(ШАҲРВАНД[ӢИ]|ШАХРВАНД[ИӢ])\b", " ", name, flags=re.I)
        name = re.sub(r"\bин\b", " ", name, flags=re.I)
        parts = [p for p in re.split(r"\s+", name.strip()) if p]
        while parts and (len(parts[-1]) <= 2 or not re.search(r"[А-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ]", parts[-1])):
            parts.pop()
        name = " ".join(parts[:4]).strip()
        if name:
            fields["name_and_surname"] = _field_template("name_and_surname", name, max(fields["name_and_surname"].get("confidence", 0.75), 0.82))

    prs = fields.get("prs_mia_rt", {}).get("value", "")
    if (("хшб" in raw_l) or ("вкд" in raw_l)) and (not prs or "вкд" not in prs.lower() or "хшб" not in prs.lower()):
        fields["prs_mia_rt"] = _field_template("prs_mia_rt", "ХШБ ВКД ҶТ", 0.82)
    elif prs and not re.search(r"(хшб|вкд|ҷт|чт)", prs, flags=re.I):
        fields["prs_mia_rt"] = _field_template("prs_mia_rt", "", 0.0)

    all_dates = []
    for block in ocr_results or []:
        d = _norm_date(str(block.get("text", "")))
        if d:
            _, y = _bbox_center(block)
            all_dates.append((y, d, float(block.get("confidence", 0.6))))
    all_dates.sort(key=lambda x: x[0])
    unique_dates = []
    for _, d, c in all_dates:
        if d not in [u[0] for u in unique_dates]:
            unique_dates.append((d, c))

    valid_until = fields.get("valid_until", {}).get("value", "")
    reg_date = fields.get("date_of_registration", {}).get("value", "")
    reg_src = fields.get("date_of_registration", {}).get("source")
    if reg_date and valid_until and reg_date == valid_until and reg_src not in {"roi", "position_date", "eval_postprocess"}:
        fields["date_of_registration"] = _field_template("date_of_registration", "", 0.0)
    if valid_until:
        earlier = [d for d, _ in unique_dates if d != valid_until]
        if earlier and (not fields.get("date_of_registration", {}).get("value") or fields["date_of_registration"]["value"] == valid_until):
            fields["date_of_registration"] = _field_template("date_of_registration", earlier[0], 0.84)
    if not fields.get("date_of_registration_extension", {}).get("value") and fields.get("date_of_registration", {}).get("value"):
        fields["date_of_registration_extension"] = _field_template("date_of_registration_extension", fields["date_of_registration"]["value"], 0.72)

    dates = [fields.get("date_of_registration", {}).get("value", ""), fields.get("valid_until", {}).get("value", "")]
    passport = fields.get("passport_number", {}).get("value", "")
    reg = fields.get("registration_card_number", {}).get("value", "")
    if reg and (_looks_like_date_or_year(reg, dates) or (passport and _digits(reg) in _digits(passport))):
        fields["registration_card_number"] = _field_template("registration_card_number", "", 0.0)
    if not fields.get("registration_card_number", {}).get("value"):
        candidates = []
        for block in ocr_results or []:
            text = str(block.get("text", "")).translate(DIGIT_FIX)
            for m in re.finditer(r"(?<!\d)(\d{6,8})(?!\d)", text):
                val = m.group(1)
                if _looks_like_date_or_year(val, dates):
                    continue
                if passport and val in _digits(passport):
                    continue
                x, y = _bbox_center(block)
                score = float(block.get("confidence", 0.5)) + (0.25 if y < 1400 else 0.0) + (0.15 if len(val) == 7 else 0.0)
                candidates.append((score, val))
        if candidates:
            candidates.sort(reverse=True)
            fields["registration_card_number"] = _field_template("registration_card_number", candidates[0][1], 0.84)

    serial = fields.get("serial_control_number", {}).get("value", "")
    reg = fields.get("registration_card_number", {}).get("value", "")
    if serial and (_looks_like_date_or_year(serial, dates) or (passport and serial in _digits(passport)) or (reg and serial in _digits(reg))):
        fields["serial_control_number"] = _field_template("serial_control_number", "", 0.0)
    if not fields.get("serial_control_number", {}).get("value"):
        candidates = []
        for block in ocr_results or []:
            text = str(block.get("text", "")).translate(DIGIT_FIX)
            for m in re.finditer(r"(?<!\d)(\d{3,5})(?!\d)", text):
                val = m.group(1)
                if val in {"2024", "2025", "2026", "2027"}:
                    continue
                if passport and val in _digits(passport):
                    continue
                if reg and val in _digits(reg):
                    continue
                if any(val in _digits(d) for d in dates if d):
                    continue
                x, y = _bbox_center(block)
                score = float(block.get("confidence", 0.5))
                if 0.45 <= (y / 3600.0) <= 0.70 and x > 900:
                    score += 0.35
                candidates.append((score, val))
        if candidates:
            candidates.sort(reverse=True)
            fields["serial_control_number"] = _field_template("serial_control_number", candidates[0][1], 0.84)

    city_match = re.search(r"(ш\.?\s*[А-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ]{3,20})", raw)
    place = fields.get("place_of_residence", {}).get("value", "")
    if city_match:
        fields["place_of_residence"] = _field_template("place_of_residence", city_match.group(1).replace("ш ", "ш. "), 0.83)
    elif place and not re.search(r"(ш\.?|ноҳия|нохия|ҷамоат|чамоат|к\.?|куча|вил\.?|вилоят)", place, flags=re.I):
        fields["place_of_residence"] = _field_template("place_of_residence", "", 0.0)

    cont = fields.get("place_of_residence_cont", {}).get("value", "")
    if cont and (re.search(r"(Момализода|Имомализода|нозир|тамдид|\d{2}\.\d{2}\.\d{4})", cont, flags=re.I) or len(cont) > 35):
        fields["place_of_residence_cont"] = _field_template("place_of_residence_cont", "", 0.0)

    return fields


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

    fields = _postprocess_like_app(fields, merged_all_ocr)
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
    ocr_handler = OCRHandler(languages=Config.OCR_LANGUAGES, fast_mode=fast_mode if fast_mode is not None else Config.OCR_FAST_MODE)
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
