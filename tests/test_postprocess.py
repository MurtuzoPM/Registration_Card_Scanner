"""Generalization tests for the card-agnostic post-processing layer.

The whole point of these tests is to prove the extractor does NOT memorise a
single card.  We feed synthetic OCR for two *different* fictional people and
assert the output reflects the input — and that none of the previously
hard-coded answers (Назарӣ / Одиназода / Момализода / Ваҳдат / Фирдавсӣ /
forced "Хитой" / the N00085561 passport hack) leak through.
"""

from ocr_engine.postprocess import postprocess_fields


def _bbox(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _block(text, y, x=600, conf=0.85):
    return {"text": text, "confidence": conf, "bbox": _bbox(x, y, x + 400, y + 60)}


def _field(value, source="roi", conf=0.8):
    return {"value": value, "confidence": conf, "label": "", "field_number": None, "source": source}


def _empty_parsed():
    keys = [
        "registration_card_number", "passport_number", "citizenship",
        "name_and_surname", "date_of_registration", "prs_mia_rt",
        "valid_until", "serial_control_number", "mia",
        "place_of_residence", "place_of_residence_cont", "inspector",
        "date_of_registration_extension",
    ]
    return {k: {"value": "", "confidence": 0.0, "label": "", "field_number": None} for k in keys}


# Values previously hard-coded into the pipeline; must never appear unprompted.
FORBIDDEN = ["Назарӣ", "Одиназода", "Момализода", "Ваҳдат", "Фирдавс", "N00085561"]


def test_card_a_fields_reflect_input():
    parsed = _empty_parsed()
    parsed["name_and_surname"] = _field("ШАҲРВАНДӢ Карим Салимов")
    parsed["citizenship"] = _field("Руссия")
    parsed["inspector"] = _field("нозир Холов Р.")
    parsed["place_of_residence"] = _field("ш. Душанбе")
    parsed["valid_until"] = _field("07.09.2026")

    ocr = [
        _block("EC1234567", 900),
        _block("1233509", 700),
        _block("ш. Душанбе", 2400),
        _block("ХШБ ВКД", 1500),
        _block("ВКД", 1900),
        _block("740", 1950, x=1200),
        _block("01.09.2025", 1700),
        _block("07.09.2026", 2000),
    ]

    out = postprocess_fields(parsed, ocr)

    assert out["name_and_surname"]["value"] == "Карим Салимов"
    assert out["citizenship"]["value"] == "Руссия"
    assert out["inspector"]["value"].startswith("Холов")
    assert out["place_of_residence"]["value"] == "ш. Душанбе"
    assert out["passport_number"]["value"] == "EC1234567"
    assert out["registration_card_number"]["value"] == "1233509"
    assert out["serial_control_number"]["value"] == "740"
    # date_of_registration is filled from the earlier date when valid_until exists
    assert out["date_of_registration"]["value"] == "01.09.2025"
    assert out["prs_mia_rt"]["value"] == "ХШБ ВКД ҶТ"
    assert out["mia"]["value"] == "ВКД"

    blob = " ".join(f["value"] for f in out.values())
    for bad in FORBIDDEN:
        assert bad not in blob, f"leaked hard-coded value: {bad}"


def test_card_b_is_different_from_card_a():
    parsed = _empty_parsed()
    parsed["name_and_surname"] = _field("ШАХРВАНДИ Зайнаб Рахимова")
    parsed["citizenship"] = _field("Афгонистон")          # spelling normalised via lexicon
    parsed["inspector"] = _field("Назаров И")
    parsed["place_of_residence"] = _field("ш. Хуҷанд")

    ocr = [
        _block("N0011801", 900),
        _block("1288203", 700),
        _block("Хуҷанд", 2400),
        _block("15.02.2024", 1700),
    ]

    out = postprocess_fields(parsed, ocr)

    assert out["name_and_surname"]["value"] == "Зайнаб Рахимова"
    assert out["citizenship"]["value"] == "Афғонистон"
    assert out["inspector"]["value"] == "Назаров И"
    assert out["place_of_residence"]["value"] == "ш. Хуҷанд"
    assert out["passport_number"]["value"] == "N0011801"
    assert out["registration_card_number"]["value"] == "1288203"

    blob = " ".join(f["value"] for f in out.values())
    for bad in FORBIDDEN:
        assert bad not in blob, f"leaked hard-coded value: {bad}"


def test_passport_n000_not_mutated():
    """The old code rewrote N0008561 -> N00085561. That hack must be gone."""
    parsed = _empty_parsed()
    out = postprocess_fields(parsed, [_block("N0008561", 900)])
    assert out["passport_number"]["value"] == "N0008561"


def test_unknown_citizenship_is_preserved_not_forced():
    parsed = _empty_parsed()
    parsed["citizenship"] = _field("Қазоқистон")  # not in lexicon -> keep as-is
    out = postprocess_fields(parsed, [_block("Қазоқистон", 1100)])
    assert out["citizenship"]["value"] == "Қазоқистон"


def test_noise_inspector_is_dropped():
    parsed = _empty_parsed()
    parsed["inspector"] = _field("а б в г д е")  # single-char noise
    out = postprocess_fields(parsed, [])
    assert out["inspector"]["value"] == ""
