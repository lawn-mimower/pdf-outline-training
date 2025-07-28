"""
PDF Outline Extractor (Hackathon Round 1A)
-----------------------------------------

Requirements satisfied:
- Accepts PDFs (≤ 50 pages) from /app/input and writes JSON to /app/output (batch mode)
- Outputs: {"title": str, "outline": [{"level": "H1|H2|H3", "text": str, "page": int}, ...]}
- Runs fully offline, CPU only, ~instant (<10s/50p on 8C/16GB)
- Optional ML model (<=200MB) via joblib. If missing, uses heuristics.
- Docker-friendly CLI.

Usage (inside container, as judges will run):
    python /app/pdf_outline_extractor.py  # auto uses /app/input -> /app/output

Manual single-file mode:
    python pdf_outline_extractor.py --input_pdf sample.pdf --output_json sample.json

Optional arguments:
    --model path/to/model.joblib   # if you have a trained classifier
    --input_dir /path/in           # default /app/input
    --output_dir /path/out         # default /app/output

"Heuristic-only" mode is recommended if you don't want to ship a model.

Author: <you>
"""

from __future__ import annotations
import argparse
import json
import os
import statistics
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise SystemExit("PyMuPDF (fitz) is required. pip install pymupdf")

# joblib is optional
try:
    import joblib  # type: ignore
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False


# ---------------------------- Utility Data Structures ---------------------------- #

class SpanBlock:
    """A consolidated text block with layout metadata."""
    __slots__ = (
        "text", "page", "font_size", "is_bold", "is_italic", "bbox", "page_width",
        "x_pos", "distance_from_prev", "is_centered", "relative_font", "dot_count",
        "starts_with_digit", "is_titlecase"
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


# ---------------------------- Core Extractor Class ---------------------------- #

class PDFOutlineExtractor:
    CENTER_TOLERANCE_FACTOR = 0.05
    MAX_FONT_DIFF_FOR_MERGE = 2.0
    VERTICAL_GAP_MULTIPLIER = 0.4
    HORIZONTAL_GAP_MULTIPLIER = 0.5
    X_POS_TOLERANCE = 8.0

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.using_model = False
        if model_path and HAS_JOBLIB and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.using_model = True
                print(f"🤖 Loaded model: {model_path}")
            except Exception as e:
                print(f"⚠️  Could not load model '{model_path}'. Falling back to heuristics. Error: {e}")
                self.model = None
        else:
            if model_path and not os.path.exists(model_path):
                print(f"⚠️  Model file not found at {model_path}. Using heuristics.")
            elif model_path and not HAS_JOBLIB:
                print("⚠️  joblib not installed. Using heuristics.")

    # ---------------------------- PDF Parsing ---------------------------- #
    def _extract_spans(self, pdf_path: str) -> List[Dict[str, Any]]:
        spans: List[Dict[str, Any]] = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Could not open PDF '{pdf_path}': {e}")
            return spans

        for page_idx, page in enumerate(doc, start=1):
            page_width = page.rect.width
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        font_name = span.get("font", "").lower()
                        bbox = list(span.get("bbox", (0, 0, 0, 0)))
                        spans.append({
                            "text": text,
                            "page": page_idx,
                            "font_size": float(span.get("size", 0)),
                            "is_bold": int("bold" in font_name),
                            "is_italic": int("italic" in font_name),
                            "bbox": bbox,
                            "x_pos": float(bbox[0]),
                            "page_width": float(page_width)
                        })
        doc.close()
        return spans

    def _consolidate(self, spans: List[Dict[str, Any]]) -> List[SpanBlock]:
        if not spans:
            return []
        spans = sorted(spans, key=lambda s: (s["page"], s["bbox"][1], s["bbox"][0]))
        merged: List[Dict[str, Any]] = [spans[0]]
        for cur in spans[1:]:
            prev = merged[-1]
            if cur["page"] != prev["page"]:
                merged.append(cur); continue
            similar_font = abs(cur["font_size"] - prev["font_size"]) <= self.MAX_FONT_DIFF_FOR_MERGE
            if not similar_font:
                merged.append(cur); continue

            # Line overlap / gaps
            v_overlap = min(prev["bbox"][3], cur["bbox"][3]) - max(prev["bbox"][1], cur["bbox"][1])
            h_gap = cur["bbox"][0] - prev["bbox"][2]
            dyn_h_gap = max(prev["font_size"], cur["font_size"]) * self.HORIZONTAL_GAP_MULTIPLIER

            should_merge = False
            if v_overlap >= -2.0 and -8.0 <= h_gap <= dyn_h_gap:
                should_merge = True
            else:
                v_gap = cur["bbox"][1] - prev["bbox"][3]
                x_diff = abs(cur["bbox"][0] - prev["bbox"][0])
                dyn_v_gap = max(prev["font_size"], cur["font_size"]) * self.VERTICAL_GAP_MULTIPLIER
                if 0 <= v_gap <= dyn_v_gap and x_diff <= self.X_POS_TOLERANCE:
                    should_merge = True

            if should_merge:
                if prev["text"] and not prev["text"].endswith(" ") and not cur["text"].startswith(" "):
                    prev["text"] += " "
                prev["text"] += cur["text"]
                # expand bbox
                prev_bbox = prev["bbox"]
                cur_bbox = cur["bbox"]
                prev_bbox[0] = min(prev_bbox[0], cur_bbox[0])
                prev_bbox[1] = min(prev_bbox[1], cur_bbox[1])
                prev_bbox[2] = max(prev_bbox[2], cur_bbox[2])
                prev_bbox[3] = max(prev_bbox[3], cur_bbox[3])
                prev["is_bold"] = prev["is_bold"] or cur["is_bold"]
            else:
                merged.append(cur)

        # Enrich features
        if merged:
            base_font = self._base_font_size(merged)
            for i, m in enumerate(merged):
                bbox = m["bbox"]
                span_center_x = (bbox[0] + bbox[2]) / 2.0
                page_center_x = m["page_width"] / 2.0
                tolerance = m["page_width"] * self.CENTER_TOLERANCE_FACTOR
                is_centered = int(abs(span_center_x - page_center_x) < tolerance)
                dist_prev = -1.0
                if i > 0 and merged[i-1]["page"] == m["page"]:
                    dist_prev = bbox[1] - merged[i-1]["bbox"][3]

                block = SpanBlock(
                    text=m["text"],
                    page=m["page"],
                    font_size=m["font_size"],
                    is_bold=int(m["is_bold"]),
                    is_italic=int(m["is_italic"]),
                    bbox=bbox,
                    page_width=m["page_width"],
                    x_pos=m["x_pos"],
                    distance_from_prev=float(dist_prev),
                    is_centered=is_centered,
                    relative_font=(m["font_size"] / base_font) if base_font else 1.0,
                    dot_count=m["text"].count('.'),
                    starts_with_digit=int(bool(m["text"]) and m["text"][0].isdigit()),
                    is_titlecase=int(m["text"].istitle())
                )
                merged[i] = block
        return merged  # type: ignore

    @staticmethod
    def _base_font_size(blocks: List[Dict[str, Any]]) -> float:
        sizes = [b["font_size"] for b in blocks if b.get("font_size")]  # type: ignore
        if not sizes:
            return 12.0
        try:
            return Counter(sizes).most_common(1)[0][0]
        except Exception:
            return statistics.median(sizes)

    # ---------------------------- Classification ---------------------------- #
    def classify_blocks(self, blocks: List[SpanBlock]) -> List[str]:
        if self.using_model and self.model is not None:
            # use model predict
            feats = [
                [
                    b.font_size, b.is_bold, b.is_italic, len(b.text), b.x_pos,
                    b.starts_with_digit, b.is_titlecase, b.dot_count, b.page,
                    b.relative_font, b.is_centered, b.distance_from_prev
                ]
                for b in blocks
            ]
            pred = self.model.predict(feats)
            # Mapping should be inside model pipeline; otherwise adjust here
            # assume labels are strings already or ints -> map
            if isinstance(pred[0], (int, float)):
                label_map = {0: 'BODY', 1: 'TITLE', 2: 'H1', 3: 'H2', 4: 'H3', 5: 'H4'}
                return [label_map.get(int(p), 'BODY') for p in pred]
            return [str(p) for p in pred]

        # Heuristic classification
        labels: List[str] = []
        for i, b in enumerate(blocks):
            txt = b.text.strip()
            rel = b.relative_font
            is_headish = (len(txt) <= 160 and '\n' not in txt and b.dot_count <= 3)
            is_shortish = len(txt.split()) <= 15
            big_gap = b.distance_from_prev > (b.font_size * 1.2)
            # Title detection candidate
            if b.page == 1 and rel >= 1.7 and b.is_centered and is_shortish:
                labels.append('TITLE'); continue
            # H1/H2/H3 based on relative size & bold/center & gap
            if rel >= 1.45 and (b.is_bold or b.is_centered or big_gap) and is_headish:
                labels.append('H1'); continue
            if rel >= 1.25 and (b.is_bold or big_gap) and is_headish:
                labels.append('H2'); continue
            if rel >= 1.12 and is_headish:
                labels.append('H3'); continue
            labels.append('BODY')
        # ensure we have exactly one TITLE; pick first H1 if none
        if 'TITLE' not in labels:
            for idx, lab in enumerate(labels):
                if lab == 'H1' and blocks[idx].page == 1:
                    labels[idx] = 'TITLE'; break
            else:
                # fallback to largest font on first page
                first_page_idxs = [i for i, b in enumerate(blocks) if b.page == 1]
                if first_page_idxs:
                    idx = max(first_page_idxs, key=lambda i: blocks[i].font_size)
                    labels[idx] = 'TITLE'
        return labels

    # ---------------------------- JSON Formatting ---------------------------- #
    @staticmethod
    def to_outline_json(blocks: List[SpanBlock], labels: List[str]) -> Dict[str, Any]:
        title_text = ""
        outline: List[Dict[str, Any]] = []
        for b, lab in zip(blocks, labels):
            if lab == 'TITLE' and not title_text:
                title_text = b.text.strip()
            elif lab in {"H1", "H2", "H3"}:
                cleaned = b.text.strip().rstrip(".:;-–—")
                if not cleaned:
                    continue
                if outline and outline[-1]["text"].lower() == cleaned.lower() and outline[-1]["page"] == b.page:
                    continue
                outline.append({"level": lab, "text": cleaned, "page": int(b.page)})
        if not title_text:
            # fallback: first H1 text or first block
            for item in outline:
                if item["level"] == "H1":
                    title_text = item["text"]
                    break
            if not title_text and blocks:
                title_text = blocks[0].text.strip()[:120]
        return {"title": title_text, "outline": outline}

    # ---------------------------- Public API ---------------------------- #
    def process_pdf(self, pdf_path: str, out_path: str):
        print(f"📄 Processing: {os.path.basename(pdf_path)}")
        spans = self._extract_spans(pdf_path)
        if not spans:
            print("  ⚠️  No spans extracted. Skipping.")
            return
        blocks = self._consolidate(spans)
        labels = self.classify_blocks(blocks)
        data = self.to_outline_json(blocks, labels)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"  ✅ Saved -> {out_path}")

    def process_folder(self, in_dir: str, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        pdfs = [f for f in os.listdir(in_dir) if f.lower().endswith('.pdf')]
        if not pdfs:
            print(f"No PDFs found in {in_dir}")
            return
        for fname in sorted(pdfs):
            inp = os.path.join(in_dir, fname)
            outp = os.path.join(out_dir, os.path.splitext(fname)[0] + '.json')
            try:
                self.process_pdf(inp, outp)
            except Exception as e:
                print(f"❌ Error on {fname}: {e}")


# ---------------------------- CLI ---------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline PDF Outline Extractor")
    p.add_argument("--input_pdf", help="Single PDF path")
    p.add_argument("--output_json", help="Single JSON output path")
    p.add_argument("--input_dir", default="/app/input", help="Batch input dir")
    p.add_argument("--output_dir", default="/app/output", help="Batch output dir")
    p.add_argument("--model", default=None, help="Optional joblib model path")
    return p.parse_args()


def main():
    args = parse_args()
    extractor = PDFOutlineExtractor(model_path=args.model)

    if args.input_pdf and args.output_json:
        extractor.process_pdf(args.input_pdf, args.output_json)
    else:
        extractor.process_folder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
