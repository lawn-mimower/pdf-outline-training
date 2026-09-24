import json
import subprocess
import sys
from pathlib import Path

import fitz
import pandas as pd

from conftest import HEADINGS, SIZE_TO_LABEL
from extractor import PDFProcessor
from fit import DocumentOutliner
import stage1
import stage2

REPO = Path(__file__).resolve().parents[1]


def pdf_spans(pdf_path):
    doc = fitz.open(pdf_path)
    spans = []
    for page_no, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    spans.append((page_no, span["text"].strip(), round(span["size"])))
    doc.close()
    return spans


def test_extract_features_gives_12_features_per_span(sample_pdf):
    spans, features = PDFProcessor().extract_features(sample_pdf)
    assert features.shape == (len(spans), 12)
    assert list(features.columns) == [f"feature_{i}" for i in range(12)]


def test_stage1_and_stage2_produce_12_features(sample_pdf, tmp_path):
    labels = []
    for page, text, size in pdf_spans(sample_pdf):
        label = SIZE_TO_LABEL[size]
        labels.append({
            "text": text, "page": page, "font_size": size, "is_bold": size > 11,
            "label": label, "label_type": "BODY_TEXT" if label == 0 else "HEADING",
            # labeller.py saves 12 features; older label files only have the 9 base ones
            "features": [float(size)] * (12 if page == 1 else 9),
        })
    labels_json = tmp_path / "Set1_target.json"
    labels_json.write_text(json.dumps(labels))

    complete = tmp_path / "complete.json"
    stage1.amend_labeled_data(sample_pdf, str(labels_json), str(complete))
    amended = json.loads(complete.read_text())
    assert len(amended) == len(labels)
    assert all(len(s["bbox"]) == 4 and s["page_width"] == 595 for s in amended)

    enriched = tmp_path / "enriched.json"
    stage2.preprocess_json_file(str(complete), str(enriched))
    spans = json.loads(enriched.read_text())
    assert {len(s["features"]) for s in spans} == {12}
    title = next(s for s in spans if s["label"] == 1)
    assert title["features"][9] == 24 / 11  # relative font size vs body text
    assert title["features"][10] == 1.0     # centred
    assert all("bbox" not in s for s in spans)


def test_train_then_predict_outline(sample_pdf, tmp_path):
    # Build a stage3-style CSV from the sample PDF's own spans, labelled by font size.
    spans, features = PDFProcessor().extract_features(sample_pdf)
    features["label"] = [SIZE_TO_LABEL[round(s["font_size"])] for s in spans]
    csv = tmp_path / "final_training_dataset.csv"
    pd.concat([features] * 10, ignore_index=True).to_csv(csv, index=False)

    model_path = tmp_path / "model.joblib"
    subprocess.run(
        [sys.executable, str(REPO / "AIH_predictor" / "prediction.py"),
         "--data", str(csv), "--model-out", str(model_path)],
        check=True, capture_output=True, cwd=tmp_path,
    )
    assert model_path.exists()

    result = DocumentOutliner(str(model_path)).process_pdf(sample_pdf)
    json.dumps(result)  # must be serialisable
    assert result["title"] == "Annual Research Report"
    level = {18: "H1", 15: "H2", 13: "H3"}
    assert result["outline"] == [
        {"level": level[size], "text": text, "page": page}
        for page, text, size, _ in HEADINGS if size != 24
    ]
