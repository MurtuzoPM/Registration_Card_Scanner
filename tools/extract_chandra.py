#!/usr/bin/env python3
"""Extract registration-card fields with the Chandra vision-LLM (experimental).

This bypasses the EasyOCR/Paddle block pipeline and asks Chandra to read the
whole card and return the 13 fields as JSON. A GPU is strongly recommended.

Usage:
  python tools/extract_chandra.py training/images/card_001.png \
      --out training/reports/card_001_chandra.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_engine.chandra_backend import ChandraFieldExtractor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Chandra end-to-end field extraction.")
    ap.add_argument("image", help="Path to a card image")
    ap.add_argument("--model", default="datalab-to/chandra", help="HF model id")
    ap.add_argument("--device", default=None, help="e.g. 'cuda', 'cpu', or 0")
    ap.add_argument("--out", default=None, help="Optional output JSON path")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = (ROOT / image_path).resolve()

    extractor = ChandraFieldExtractor(model_name=args.model, device=args.device)
    fields = extractor.extract_fields(str(image_path))

    print(json.dumps(fields, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = (ROOT / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(fields, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
