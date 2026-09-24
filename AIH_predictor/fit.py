import fitz # PyMuPDF
import pandas as pd
from collections import Counter
import os
import joblib
import json

class PDFProcessor:
    """
    Step 1: Processes a new PDF file, extracts text spans, and engineers
    the 12 features required for model prediction.
    """
    def __init__(self):
        self.CENTER_TOLERANCE_FACTOR = 0.05

    def _get_base_font_size(self, spans):
        if not spans: return 12.0
        font_sizes = [s['font_size'] for s in spans if s.get('font_size')]
        if not font_sizes: return 12.0
        return Counter(font_sizes).most_common(1)[0][0]

    def extract_features(self, pdf_path):
        if not os.path.exists(pdf_path):
            print(f"❌ Error: PDF file not found at {pdf_path}")
            return None, None
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error: Could not open or process PDF '{pdf_path}': {e}")
            return None, None

        all_spans_info = []
        for page_num, page in enumerate(doc):
            page_width = page.rect.width
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block: continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text: continue
                        all_spans_info.append({
                            "text": text, "page": page_num + 1, "font_size": span["size"],
                            "is_bold": "Bold" in span["font"], "is_italic": "Italic" in span["font"],
                            "x_pos": span["bbox"][0], "bbox": span["bbox"], "page_width": page_width
                        })

        if not all_spans_info: return None, None
        spans_sorted = sorted(all_spans_info, key=lambda s: (s['page'], s['bbox'][1]))
        base_font_size = self._get_base_font_size(spans_sorted)
        final_features_list = []
        for i, span in enumerate(spans_sorted):
            initial_features = [
                span["font_size"], int(span["is_bold"]), int(span["is_italic"]), len(span["text"]),
                span["x_pos"], int(span["text"][0].isdigit() if span["text"] else 0),
                int(span["text"].istitle()), span["text"].count('.'), span["page"]
            ]
            
            relative_font_size = span["font_size"] / base_font_size if base_font_size > 0 else 1.0
            span_center_x = (span["bbox"][0] + span["bbox"][2]) / 2.0
            page_center_x = span["page_width"] / 2.0
            is_centered = 1.0 if abs(span_center_x - page_center_x) < (span["page_width"] * self.CENTER_TOLERANCE_FACTOR) else 0.0
            distance_from_previous = span['bbox'][1] - spans_sorted[i-1]['bbox'][3] if i > 0 and span['page'] == spans_sorted[i-1]['page'] else -1.0
            
            final_features_list.append(initial_features + [relative_font_size, is_centered, distance_from_previous])

        feature_names = [f"feature_{i}" for i in range(12)]
        features_df = pd.DataFrame(final_features_list, columns=feature_names)
        return spans_sorted, features_df

class DocumentOutliner:
    """
    Orchestrates the full pipeline: feature extraction, prediction,
    and final output formatting.
    """
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}.")
        print(f"🧠 Loading model from '{model_path}'...")
        self.model = joblib.load(model_path)
        self.processor = PDFProcessor()
        self.label_map = {
            0: "BODY_TEXT", 1: "TITLE", 2: "H1",
            3: "H2", 4: "H3", 5: "H4"
        }
        print("✅ Model loaded successfully.")

    def _format_output(self, labeled_spans, original_pdf_name):
        """
        Step 3: Formats the predicted labels into the final competition JSON schema.
        """
        output = {
            "pdf_name": original_pdf_name,
            "title": "N/A",
            "outline": []
        }

        # First pass to find the title
        for span in labeled_spans:
            if span.get('predicted_label') == 'TITLE':
                output['title'] = span['text']
                break 

        # Second pass to build the outline
        for span in labeled_spans:
            label = span.get('predicted_label', '')
            if label in ("H1", "H2", "H3"):
                output['outline'].append({
                    "level": label,
                    "text": span['text'],
                    "page": span['page']
                })
        return output

    def process_pdf(self, pdf_path):
        """
        Main method to run the entire pipeline on a single PDF.
        """
        print("\n--- Running Step 1: Feature Extraction ---")
        original_spans, features_df = self.processor.extract_features(pdf_path)
        if features_df is None:
            print("Aborting: Feature extraction failed.")
            return None

        print("\n--- Running Step 2: Predicting Labels ---")
        predictions = self.model.predict(features_df)
        for i, span in enumerate(original_spans):
            predicted_label_num = predictions[i]
            span['predicted_label'] = self.label_map.get(predicted_label_num, "UNKNOWN")
        print("✅ Prediction complete.")

        print("\n--- Running Step 3: Formatting to Final JSON ---")
        pdf_name = os.path.basename(pdf_path)
        final_json_output = self._format_output(original_spans, pdf_name)
        print("✅ JSON output formatted.")
        return final_json_output

# --- HOW TO USE THE FINAL SCRIPT ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict the title/H1-H3 outline of a PDF with a trained model.")
    parser.add_argument("pdf", help="PDF to process")
    parser.add_argument("--model", default="lgbm_document_model_final.joblib", help="Model saved by prediction.py")
    parser.add_argument("--output", default="output_outline.json", help="Where to write the outline JSON")
    args = parser.parse_args()
    MODEL_PATH = args.model
    NEW_PDF_PATH = args.pdf
    OUTPUT_JSON_PATH = args.output
    
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: The model file '{MODEL_PATH}' was not found.")
        print("Please run the final, corrected model training script (`prediction.py`) to create it.")
    elif not os.path.exists(NEW_PDF_PATH):
        print(f"FATAL ERROR: The PDF to process, '{NEW_PDF_PATH}', was not found.")
    else:
        outliner = DocumentOutliner(model_path=MODEL_PATH)
        final_output = outliner.process_pdf(pdf_path=NEW_PDF_PATH)
        if final_output:
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=4)
            print(f"\n🎉 Success! Final outline saved to '{OUTPUT_JSON_PATH}'")
            print("\n--- Output Preview ---")
            print(json.dumps(final_output, indent=2))
