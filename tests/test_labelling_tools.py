import json

import pandas as pd

import labeler
from labeller import PDFLabeler


def test_pdf_labeller_saves_enriched_12_feature_labels(sample_pdf, tmp_path, monkeypatch):
    # title, 3 body lines (one skipped), first heading, then quit
    answers = iter(["1", "0", "s", "0", "2", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    out = tmp_path / "training_data_enriched.json"

    lab = PDFLabeler(str(out))
    assert lab.load_pdf(sample_pdf)
    lab.start_labeling_session()

    saved = json.loads(out.read_text())
    assert [s["label_type"] for s in saved] == ["TITLE", "BODY_TEXT", "BODY_TEXT", "H1"]
    assert saved[0]["text"] == "Annual Research Report"
    assert {len(s["features"]) for s in saved} == {12}

    # A second session skips spans that are already labelled.
    again = PDFLabeler(str(out))
    again.load_pdf(sample_pdf)
    assert "Annual Research Report" not in [s["text"] for s in again.current_pdf_spans]


def test_csv_labeler_labels_undo_and_resume(tmp_path, monkeypatch):
    src = tmp_path / "spans.csv"
    pd.DataFrame({
        "text": ["Title", "Intro", "Body"], "font_size": [20.0, 14.0, 10.0],
        "is_bold": [1, 1, 0], "page": [1, 1, 1], "is_centered": [1, 0, 0],
    }).to_csv(src, index=False)
    out = tmp_path / "labelled.csv"
    monkeypatch.setattr(labeler.os, "system", lambda cmd: 0)

    answers = iter(["1", "5", "u", "2", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    labeler.CSVLabeler(str(src), str(out)).run()
    assert pd.read_csv(out)["label"].tolist() == ["TITLE", "H1", "UNLABELED"]

    answers = iter(["table"])
    labeler.CSVLabeler(str(src), str(out)).run()
    assert pd.read_csv(out)["label"].tolist() == ["TITLE", "H1", "table"]
