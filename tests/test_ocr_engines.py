"""Tests for the pluggable OCR engine layer (no models required).

These exercise the pure result-parsing / prompt / mapping logic and the
factory selection, so they run without PaddleOCR, transformers, or any model
download.
"""

import pytest

from ocr_engine.paddle_backend import PaddleOCRHandler
from ocr_engine.chandra_backend import build_prompt, parse_model_json, map_to_fields
from ocr_engine.factory import create_ocr_handler


# --------------------------- PaddleOCR parsing ---------------------------
def test_paddle_parse_2x_list_shape():
    res = [[
        [[[10, 20], [100, 20], [100, 50], [10, 50]], ("Назарӣ Салмо", 0.93)],
        [[[10, 60], [80, 60], [80, 90], [10, 90]], ("740", 0.98)],
    ]]
    parsed = PaddleOCRHandler.parse_result(res)
    assert parsed[0] == ([[10, 20], [100, 20], [100, 50], [10, 50]], "Назарӣ Салмо", 0.93)
    assert parsed[1][1] == "740" and parsed[1][2] == 0.98


def test_paddle_parse_3x_dict_polys():
    res = [{
        "rec_texts": ["abc", "123"],
        "rec_scores": [0.9, 0.8],
        "rec_polys": [
            [[1, 2], [3, 2], [3, 4], [1, 4]],
            [[5, 6], [7, 6], [7, 8], [5, 8]],
        ],
    }]
    parsed = PaddleOCRHandler.parse_result(res)
    assert [p[1] for p in parsed] == ["abc", "123"]
    assert parsed[0][0] == [[1, 2], [3, 2], [3, 4], [1, 4]]


def test_paddle_parse_3x_rec_boxes_xyxy():
    res = [{"rec_texts": ["x"], "rec_scores": [0.7], "rec_boxes": [[10, 20, 110, 60]]}]
    parsed = PaddleOCRHandler.parse_result(res)
    assert parsed[0][0] == [[10, 20], [110, 20], [110, 60], [10, 60]]


def test_paddle_parse_empty():
    assert PaddleOCRHandler.parse_result(None) == []
    assert PaddleOCRHandler.parse_result([]) == []


# --------------------------- Chandra pure logic ---------------------------
def test_chandra_prompt_lists_all_keys():
    prompt = build_prompt()
    for key in ["registration_card_number", "passport_number", "name_and_surname",
                "inspector", "date_of_registration_extension"]:
        assert key in prompt


def test_chandra_parse_json_fenced():
    raw = 'Here you go:\n```json\n{"name_and_surname": "Карим Салимов"}\n```\nThanks'
    assert parse_model_json(raw) == {"name_and_surname": "Карим Салимов"}


def test_chandra_parse_json_plain():
    raw = 'noise {"citizenship": "Афғонистон", "serial_control_number": "740"} more'
    data = parse_model_json(raw)
    assert data["citizenship"] == "Афғонистон"
    assert data["serial_control_number"] == "740"


def test_chandra_parse_json_invalid():
    assert parse_model_json("no json at all") == {}
    assert parse_model_json("") == {}


def test_chandra_map_to_fields():
    fields = map_to_fields({"name_and_surname": "Карим", "passport_number": "P05661035"})
    assert fields["name_and_surname"]["value"] == "Карим"
    assert fields["name_and_surname"]["confidence"] > 0
    assert fields["passport_number"]["value"] == "P05661035"
    # absent field present but empty
    assert fields["inspector"]["value"] == ""
    assert fields["inspector"]["confidence"] == 0.0
    # every field carries the chandra source tag
    assert all(f["source"] == "chandra" for f in fields.values())


# --------------------------- Factory selection ---------------------------
def test_factory_chandra_rejected_as_block_engine():
    with pytest.raises(ValueError):
        create_ocr_handler(engine="chandra")


def test_factory_paddle_requires_package():
    # PaddleOCR isn't installed in CI; the factory should surface a clear error
    # rather than silently falling back.
    with pytest.raises(RuntimeError):
        create_ocr_handler(engine="paddleocr")
