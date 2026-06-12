#!/usr/bin/env python3
"""Calibrate a fixed-layout ROI template from labelled debug reports.

For every labelled card we already have a ``*_debug.json`` report containing the
raw OCR blocks (text + pixel bounding box) and the processed ``image_size``.
This script matches each ground-truth field value to the OCR block(s) it came
from, converts the boxes to *relative* coordinates (0..1) and aggregates them
across all cards into ``ocr_engine/data/card_template.json``.

It never needs the original images - only the debug JSON + labels.

Usage:
  python tools/calibrate_template.py \
      --labels training/labels.local.json \
      --reports training/reports \
      --out ocr_engine/data/card_template.json
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Vertical order of the fields on the card (used to interpolate y-bands for
# fields the OCR never managed to read on any card).
FIELD_ORDER = [
    "registration_card_number",
    "passport_number",
    "citizenship",
    "name_and_surname",
    "date_of_registration",
    "prs_mia_rt",
    "serial_control_number",
    "valid_until",
    "mia",
    "place_of_residence",
    "place_of_residence_cont",
    "inspector",
    "date_of_registration_extension",
]

# Horizontal padding (relative) added around located boxes.
PAD_X = 0.04
PAD_Y = 0.02


def _norm(s: str) -> str:
    s = (s or "").lower()
    table = {"ё": "е", "ӯ": "у", "ӣ": "и", "ҳ": "х", "ҷ": "ч", "қ": "к", "ғ": "г"}
    for a, b in table.items():
        s = s.replace(a, b)
    return s


def _compact(s: str) -> str:
    return re.sub(r"[^a-zа-я0-9]", "", _norm(s))


def _alpha_needles(value: str):
    return [_compact(w) for w in re.findall(r"[А-Яа-яЁёӢӣӮӯҚқҒғҲҳҶҷ]{4,}", value)]


def _digit_needles(value: str):
    return [d for d in re.findall(r"\d{3,}", value)]


def _bbox_rel(bbox, w, h):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return [min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h]


def _union(boxes):
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


NUMERIC_FIELDS = {"registration_card_number", "serial_control_number", "passport_number"}
DATE_FIELDS = {"date_of_registration", "valid_until", "date_of_registration_extension"}
_DATE_RE = re.compile(r"(\d{1,2})\D+(\d{1,2})\D+(\d{2,4})")


def _match_blocks(field, value, blocks, w, h):
    """Return the relative union box of OCR blocks that match *value*.

    Field-aware so a short serial (``123``) cannot match a long registration
    number (``1233509``), and different dates are not collapsed together.
    """
    hits = []

    if field in DATE_FIELDS:
        m = _DATE_RE.search(value)
        if not m:
            return None
        dd, mm = m.group(1).zfill(2), m.group(2).zfill(2)
        for b in blocks:
            bm = _DATE_RE.search(b.get("text", ""))
            if bm and bm.group(1).zfill(2) == dd and bm.group(2).zfill(2) == mm and b.get("bbox"):
                hits.append(_bbox_rel(b["bbox"], w, h))
        return _union(hits) if hits else None

    if field in NUMERIC_FIELDS:
        vdig = re.sub(r"\D", "", value)
        if len(vdig) < 3:
            return None
        for b in blocks:
            ddig = re.sub(r"\D", "", b.get("text", ""))
            if len(ddig) < 3 or not b.get("bbox"):
                continue
            ratio = SequenceMatcher(None, vdig, ddig).ratio()
            if ddig == vdig or (ratio >= 0.8 and abs(len(ddig) - len(vdig)) <= 2):
                hits.append(_bbox_rel(b["bbox"], w, h))
        return _union(hits) if hits else None

    # Alpha fields (name, citizenship, place, inspector, ...)
    alpha = _alpha_needles(value)
    if not alpha:
        return None
    for b in blocks:
        ctext = _compact(b.get("text", ""))
        if not ctext or not b.get("bbox"):
            continue
        if any(n in ctext or (len(ctext) >= 5 and ctext in n) for n in alpha):
            hits.append(_bbox_rel(b["bbox"], w, h))
    return _union(hits) if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="training/labels.local.json")
    ap.add_argument("--reports", default="training/reports")
    ap.add_argument("--out", default="ocr_engine/data/card_template.json")
    args = ap.parse_args()

    labels = json.loads((ROOT / args.labels).read_text(encoding="utf-8"))
    reports_dir = ROOT / args.reports

    located: dict[str, list[list[float]]] = {}
    coverage: dict[str, int] = {}
    n_cards = 0

    for sample in labels.get("samples", []):
        stem = Path(sample["image"]).stem
        report_path = reports_dir / f"{stem}_debug.json"
        if not report_path.exists():
            print(f"!! no report for {stem} ({report_path.name}) - skipping")
            continue
        n_cards += 1
        report = json.loads(report_path.read_text(encoding="utf-8"))
        w, h = report.get("image_size") or [None, None]
        if not w or not h:
            print(f"!! {stem}: missing image_size - skipping")
            continue
        blocks = report.get("raw_ocr", [])
        print(f"\n=== {stem}  ({w}x{h}, {len(blocks)} blocks) ===")
        for field, value in sample.get("fields", {}).items():
            if not value:
                continue
            box = _match_blocks(field, value, blocks, w, h)
            if box:
                located.setdefault(field, []).append(box)
                coverage[field] = coverage.get(field, 0) + 1
                print(f"  [ok] {field:32s} {value!r}")
                print(f"       box = [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]")
            else:
                print(f"  [--] {field:32s} {value!r}  (not found in OCR)")

    # Aggregate located boxes into a template, padded.
    template: dict[str, list[float]] = {}
    bands: dict[str, tuple[float, float]] = {}
    for field, boxes in located.items():
        agg = _union(boxes)
        agg = [
            max(0.0, agg[0] - PAD_X),
            max(0.0, agg[1] - PAD_Y),
            min(1.0, agg[2] + PAD_X),
            min(1.0, agg[3] + PAD_Y),
        ]
        template[field] = [round(v, 3) for v in agg]
        bands[field] = (agg[1], agg[3])

    # Interpolate y-bands for fields never located, using neighbours in
    # FIELD_ORDER; keep a wide horizontal span so OCR still has a chance.
    for i, field in enumerate(FIELD_ORDER):
        if field in template:
            continue
        prev_band = next((bands[FIELD_ORDER[j]] for j in range(i - 1, -1, -1)
                          if FIELD_ORDER[j] in bands), None)
        next_band = next((bands[FIELD_ORDER[j]] for j in range(i + 1, len(FIELD_ORDER))
                          if FIELD_ORDER[j] in bands), None)
        if prev_band and next_band:
            y1 = prev_band[1]
            y2 = next_band[0]
        elif prev_band:
            y1 = prev_band[1]
            y2 = min(1.0, y1 + 0.08)
        elif next_band:
            y2 = next_band[0]
            y1 = max(0.0, y2 - 0.08)
        else:
            continue
        if y2 <= y1:
            y2 = min(1.0, y1 + 0.06)
        template[field] = [0.18, round(max(0.0, y1 - 0.01), 3),
                           0.97, round(min(1.0, y2 + 0.01), 3)]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ COVERAGE ================")
    for field in FIELD_ORDER:
        c = coverage.get(field, 0)
        tag = "data" if field in coverage else "interp"
        print(f"  {field:32s} {c}/{n_cards}  ({tag})")
    print(f"\nTemplate written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
