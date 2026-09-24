# pdf-outline-training

Round 1A of the Adobe India Hackathon 2025 ("Connecting the Dots") asked for a tool that reads a PDF
(up to 50 pages) and returns its title and H1–H3 headings as JSON, running offline on a CPU. Heading
styles vary from one document to the next, so this repository treats the task as supervised
classification of text spans. You hand-label the spans of real PDFs, describe each span with 12 layout
features, and train a LightGBM classifier that has to cope with a label set that is mostly body text.
This is the training side of the project. The submitted extractor is in
[lawn-mimower/Adobe-1A](https://github.com/lawn-mimower/Adobe-1A).

## How it works

```
labeller.py (interactive)  or  SetN_feature.pdf + SetN_target.json pairs
        |
AIH_preprocessor/stage1.py      match each labelled span to the PDF, add bbox and page width
AIH_preprocessor/stage2.py      add relative font size, centring, vertical gap -> 12 features
AIH_preprocessor/stage3.ipynb   merge all documents -> final_training_dataset.csv
        |
AIH_model/LGBM_balancer.ipynb               compare class_weight / SMOTE / ADASYN
AIH_model/RandomizedSearchCV_HPtuning.ipynb tune a SMOTE + LightGBM pipeline
AIH_predictor/prediction.py                 train the final model on every row
AIH_predictor/fit.py                        outline JSON for a new PDF
        |
pdf_outline_extractor.py + Dockerfile       standalone extractor: heuristics, or --model
```

Each text span (as PyMuPDF extracts it) becomes one row with these features:

| # | Feature | # | Feature |
|---|---|---|---|
| 0 | font size | 6 | text is title case |
| 1 | bold (from font name) | 7 | number of `.` characters |
| 2 | italic (from font name) | 8 | page number |
| 3 | text length | 9 | font size ÷ the document's most common body font size |
| 4 | x position | 10 | centred (within 5% of page width) |
| 5 | starts with a digit | 11 | vertical gap to the previous span on the page (−1 for the first) |

The labels are `0 BODY_TEXT`, `1 TITLE`, `2 H1`, `3 H2`, `4 H3` and `5 H4`. H4 is labelled and
predicted, but it never appears in the output outline.

The standalone extractor first merges fragmented spans into blocks. A span joins the previous one when
the font sizes differ by at most 2 pt and it either continues the same line or starts at the same left
edge less than 0.4 × the font size below it. The extractor then labels each block. Without a model it
uses font-size ratios against the document's most common size:

| Label | Rule |
|---|---|
| TITLE | Page 1, ratio ≥ 1.7, centred, 15 words or fewer |
| H1 | Ratio ≥ 1.45, plus bold, centred or a large gap above |
| H2 | Ratio ≥ 1.25, plus bold or a large gap above |
| H3 | Ratio ≥ 1.12 |

Heading candidates must be 160 characters or shorter with at most three dots. With `--model` the
extractor uses a joblib classifier trained on the same 12 features instead. The output uses the
challenge schema:

```json
{"title": "Annual Research Report",
 "outline": [{"level": "H1", "text": "1 Introduction", "page": 1},
             {"level": "H2", "text": "1.1 Background", "page": 1}]}
```

## Training data (not included)

The labelled PDFs are not in the repository. `stage1.py --data-dir DIR` pairs files by name:
`SetN_feature[_XX].pdf` goes with `SetN_target[_XX].json`, where `_XX` is an optional language tag
such as `_JAP`. Each target file is a JSON list of labelled spans in the format that `labeller.py`
writes:

```json
[{"text": "2. Project Goals", "page": 1, "font_size": 14.0, "is_bold": true,
  "features": [14.0, 1, 0, 16, 72.0, 1, 1, 1, 1], "label": 2, "label_type": "H1"}]
```

Only the first nine features are needed: font size to page number. `stage2.py` recomputes the last three.
Stage 1 matches each labelled span to the PDF by page and stripped text, and drops any it cannot find
with a warning.

The author's set has 13 pairs: 9 in English and one each in Japanese, German, Spanish and French.
After matching, it gives 2,422 labelled spans: 1,937 BODY_TEXT, 21 TITLE, 82 H1, 163 H2, 156 H3 and
63 H4. One 62-page document supplies 1,719 of those spans.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt

cd AIH_preprocessor
# 1a. Label interactively: put PDFs in source_pdfs/, then answer 0-5 / s / q for each span.
#     The output already has all 12 features, so it goes straight into labeled_data_enriched/.
python labeller.py                                   # -> training_data_enriched.json
mkdir -p labeled_data_enriched && cp training_data_enriched.json labeled_data_enriched/
# 1b. ...or start from SetN_feature.pdf / SetN_target.json pairs
python stage1.py --data-dir /path/to/pairs           # -> labeled_data_complete/
python stage2.py                                     # -> labeled_data_enriched/

# 2. One CSV from every file in labeled_data_enriched/
jupyter nbconvert --to notebook --execute --output-dir ../runs stage3.ipynb   # -> final_training_dataset.csv

# 3. Train and predict
python ../AIH_predictor/prediction.py                # SMOTE + LightGBM -> lgbm_document_model_final.joblib
python ../AIH_predictor/fit.py some.pdf --model lgbm_document_model_final.joblib   # -> output_outline.json
cd ..

# 4. Experiments (the notebooks read ../AIH_preprocessor/final_training_dataset.csv by default)
BALANCING=adasyn jupyter nbconvert --to notebook --execute --output-dir runs AIH_model/LGBM_balancer.ipynb
jupyter nbconvert --to notebook --execute --output-dir runs AIH_model/RandomizedSearchCV_HPtuning.ipynb
```

- `prediction.py` oversamples with SMOTE's default of 5 neighbours, so every label needs at least 6
  spans.
- `BALANCING` takes `class_weight` (LightGBM `is_unbalance=True`), `smote` (the default) or
  `adasyn`. Both oversamplers resample the training split only.
- `DATASET_PATH` points either notebook at another CSV.
- `labeler.py` in the repository root is a separate CSV labelling tool with undo and resume. It
  expects consolidated spans with `text`, `font_size`, `is_bold`, `page` and `is_centered` columns.

### Standalone extractor

```bash
pip install -r requirements.txt
python pdf_outline_extractor.py --input_pdf doc.pdf --output_json doc.json           # heuristics
python pdf_outline_extractor.py --input_dir pdfs/ --output_dir out/ \
    --model AIH_preprocessor/lgbm_document_model_final.joblib                        # trained model

docker build -t pdf-outline .
docker run --rm -v "$PWD/input:/app/input:ro" -v "$PWD/output:/app/output" --network none pdf-outline
```

Without arguments the script processes every PDF in `/app/input` and writes `<name>.json` to
`/app/output`. The image runs heuristics only. To ship a model, copy it in (see the commented `COPY`
line in the `Dockerfile`) and append `--model /app/<file>.joblib` to `docker run`. If the model file is
missing or fails to load, the extractor falls back to heuristics.

### Alternative notebook

`new_smote.ipynb` trains on a CSV with named feature columns plus `label` and `label_name`
(`DATASET_PATH`, default `clean_training_dataset.csv`). No script in the repository writes that CSV.
You can build it from stage 3's `feature_0`…`feature_11` columns in the order of the table above. The
notebook imputes `-1` labels with a random forest, applies SMOTE and then tests the model on
`TEST_PDF_PATH`.

## Tests

```bash
pytest          # 13 offline tests
```

The tests generate their own PDFs and data, so they need neither the labelled set nor network access.
They cover the heuristic extractor end to end, batch mode, the mapping from integer predictions to
labels, the fallback when a model is missing, both labelling tools (with scripted input), 12-feature
extraction, stages 1 and 2, a train-then-predict round trip, and `LGBM_balancer.ipynb` with each
`BALANCING` option.

## Results and status

The pipeline was run end to end on the author's 13 labelled PDFs. The notebooks use a stratified
80/20 split of spans (485 test spans, only 4 of them TITLE):

| Notebook | Metric | Value |
|---|---|---|
| `LGBM_balancer`, `class_weight` | Test accuracy | 0.975 |
| `LGBM_balancer`, `smote` | Test accuracy | 0.979 |
| `LGBM_balancer`, `adasyn` | Test accuracy | 0.979 |
| `RandomizedSearchCV_HPtuning` | Best 3-fold CV weighted F1 on the training split | 0.967 |

Treat these numbers as a sanity check on a very small dataset, not as a measure of how well the
model generalises:

- The split is by span, not by document, so every test document also contributes training spans.
- Most rows are body text, and 71% of the spans come from one document.
- In one document, the numeric labels that training uses disagree with the label names on about 110
  spans.

The heuristics-only Docker image was built and run with `--network none` on sample PDFs.

## Known limitations

- **Headings lost in merging:** block merging runs in both extractor modes. A heading whose font is
  within 2 pt of the body text, with the first body line starting at the same left edge just below
  it, gets merged into that line and drops out of the outline. For example, a 13 pt bold heading
  over 11 pt text at a normal line spacing is lost.
- **Training and inference differ:** training features come from raw PyMuPDF spans, and the base font
  size comes from spans labelled BODY_TEXT. The extractor computes features on merged blocks and takes
  the most common size overall as the base. A trained model therefore sees slightly different inputs
  at inference. On the synthetic two-page PDF from the tests, heuristics recover the intended H1/H2/H3
  levels, while the model trained on the 13 documents labels the H2 headings as H1.
- **Page limit not enforced:** the extractor does not check the 50-page limit.
- **Two output variants:** `fit.py` adds `pdf_name` and returns `"N/A"` when it finds no title, while
  `pdf_outline_extractor.py` falls back to the first H1 or the first block.

## Repository layout

| Path | Contents |
|---|---|
| `AIH_preprocessor/` | `labeller.py` (interactive span labelling), `stage1.py`, `stage2.py`, `stage3.ipynb` |
| `AIH_model/` | `LGBM_balancer.ipynb` (`BALANCING` option), `RandomizedSearchCV_HPtuning.ipynb` |
| `AIH_predictor/` | `prediction.py` (train), `fit.py` (predict an outline), `extractor.py` (feature extraction demo) |
| `new_smote.ipynb` | Alternative training notebook (see above) |
| `labeler.py` | CSV labelling tool |
| `pdf_outline_extractor.py`, `Dockerfile`, `requirements.txt` | Standalone extractor |
| `requirements-train.txt`, `pytest.ini`, `tests/` | Training dependencies and tests |

Licence: MIT — see [LICENSE](LICENSE).
