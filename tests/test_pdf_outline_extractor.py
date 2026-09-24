import json

import numpy as np

from conftest import HEADINGS
from pdf_outline_extractor import PDFOutlineExtractor


def test_heuristic_outline_end_to_end(sample_pdf, tmp_path):
    out = tmp_path / "sample.json"
    PDFOutlineExtractor().process_pdf(sample_pdf, str(out))
    data = json.loads(out.read_text())

    assert data["title"] == "Annual Research Report"
    assert data["outline"] == [
        {"level": "H1", "text": "1 Introduction", "page": 1},
        {"level": "H2", "text": "1.1 Background", "page": 1},
        {"level": "H3", "text": "1.1.1 Scope", "page": 1},
        {"level": "H1", "text": "2 Methods", "page": 2},
        {"level": "H2", "text": "2.1 Data Sources", "page": 2},
    ]


def test_batch_mode_writes_one_json_per_pdf(sample_pdf, tmp_path):
    out_dir = tmp_path / "out"
    PDFOutlineExtractor().process_folder(str(tmp_path), str(out_dir))
    data = json.loads((out_dir / "sample.json").read_text())
    assert set(data) == {"title", "outline"}


class FontSizeModel:
    """Stand-in classifier returning numpy integer labels like LightGBM does."""

    def predict(self, feats):
        size_to_label = {24: 1, 18: 2, 15: 3, 13: 4}
        return np.array([size_to_label.get(round(f[0]), 0) for f in feats], dtype=np.int64)


def test_model_integer_predictions_are_mapped_to_levels(sample_pdf, tmp_path):
    extractor = PDFOutlineExtractor()
    extractor.model, extractor.using_model = FontSizeModel(), True
    out = tmp_path / "sample.json"
    extractor.process_pdf(sample_pdf, str(out))
    data = json.loads(out.read_text())

    assert data["title"] == "Annual Research Report"
    expected = [(t, p) for p, t, size, _ in HEADINGS if size != 24]
    assert [(h["text"], h["page"]) for h in data["outline"]] == expected
    assert [h["level"] for h in data["outline"]] == ["H1", "H2", "H3", "H1", "H2"]


def test_missing_model_falls_back_to_heuristics(tmp_path):
    extractor = PDFOutlineExtractor(model_path=str(tmp_path / "missing.joblib"))
    assert extractor.using_model is False
