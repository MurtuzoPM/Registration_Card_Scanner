"""Test bootstrap.

The OCR/image dependencies (cv2, numpy, PIL) are not needed to test the pure
parsing/post-processing logic, and may be absent in CI.  Stub them so the
``ocr_engine`` package imports cleanly, and make the repo root importable.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _name in ("cv2", "numpy"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)

if "PIL" not in sys.modules:
    _pil = types.ModuleType("PIL")
    _pil.Image = types.SimpleNamespace()
    _pil.ImageOps = types.SimpleNamespace()
    sys.modules["PIL"] = _pil
