import fitz  # PyMuPDF
import pandas as pd
from collections import Counter
import os

class PDFProcessor:
    """
    A class to process a new PDF file, extract text spans, and engineer
    features ready for model prediction.
    """
    def __init__(self):
        # A tolerance for checking if a span is centered.
        self.CENTER_TOLERANCE_FACTOR = 0.05
    
    def _get_base_font_size(self, spans):
        """
        Calculates the most common font size from a list of spans.
        In prediction, we don't have 'BODY_TEXT' labels, so we use the
        most common font size overall as a heuristic.
        """
        if not spans:
            return 12.0 # A reasonable default

        font_sizes = [s['font_size'] for s in spans if s.get('font_size')]
        if not font_sizes:
            return 12.0

        return Counter(font_sizes).most_common(1)[0][0]

    def extract_features(self, pdf_path):
        """
        Main method to take a PDF path and return model-ready features.
        """
        if not os.path.exists(pdf_path):
            print(f"❌ Error: PDF file not found at {pdf_path}")
            return None, None

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error: Could not open or process PDF '{pdf_path}': {e}")
            return None, None

        all_spans_info = []
        
        # --- 1. Initial Raw Feature Extraction ---
        print(f"📄 Reading and extracting raw spans from '{pdf_path}'...")
        for page_num, page in enumerate(doc):
            page_width = page.rect.width
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block: continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text: continue
                        
                        # Store raw info and initial features
                        all_spans_info.append({
                            "text": text,
                            "page": page_num + 1,
                            "font_size": span["size"],
                            "is_bold": "Bold" in span["font"],
                            "is_italic": "Italic" in span["font"],
                            "x_pos": span["bbox"][0],
                            "bbox": span["bbox"],
                            "page_width": page_width
                        })

        if not all_spans_info:
            print("⚠️ Warning: No text spans were extracted from the document.")
            return None, None
            
        # --- 2. Advanced Feature Engineering ---
        print("⚙️ Engineering advanced features...")
        
        # Sort spans by reading order (page, then vertical position)
        spans_sorted = sorted(all_spans_info, key=lambda s: (s['page'], s['bbox'][1]))
        
        # Calculate a single base font size for the whole document
        base_font_size = self._get_base_font_size(spans_sorted)
        print(f"  - Calculated base font size for document: {base_font_size:.2f}pt")

        final_features_list = []
        for i, span in enumerate(spans_sorted):
            # --- A. Basic Features (calculated again in order) ---
            initial_features = [
                span["font_size"],
                int(span["is_bold"]),
                int(span["is_italic"]),
                len(span["text"]),
                span["x_pos"],
                int(span["text"][0].isdigit() if span["text"] else 0),
                int(span["text"].istitle()),
                span["text"].count('.'),
                span["page"]
            ]
            
            # --- B. Normalized and Contextual Features ---
            relative_font_size = span["font_size"] / base_font_size if base_font_size > 0 else 1.0
            
            span_center_x = (span["bbox"][0] + span["bbox"][2]) / 2.0
            page_center_x = span["page_width"] / 2.0
            tolerance = span["page_width"] * self.CENTER_TOLERANCE_FACTOR
            is_centered = 1.0 if abs(span_center_x - page_center_x) < tolerance else 0.0
            
            distance_from_previous = -1.0 # Default value
            if i > 0:
                prev_span = spans_sorted[i-1]
                if span['page'] == prev_span['page']:
                    distance_from_previous = span['bbox'][1] - prev_span['bbox'][3]
            
            # Combine all features into the final 12-feature vector
            final_features = initial_features + [relative_font_size, is_centered, distance_from_previous]
            final_features_list.append(final_features)
            
        # --- 3. Create Final DataFrame ---
        feature_names = [f"feature_{i}" for i in range(12)]
        features_df = pd.DataFrame(final_features_list, columns=feature_names)
        
        print(f"✅ Feature extraction complete. Found {len(features_df)} spans.")
        
        # Return the original span info (we'll need it later) and the feature DataFrame
        return spans_sorted, features_df


# --- Example of How to Use the Class ---
if __name__ == "__main__":
    # Path to a new, unseen PDF you want to process
    # Make sure you have a PDF here to test the script
    NEW_PDF_PATH = "test_sample.pdf" # Replace with your test PDF

    processor = PDFProcessor()
    
    # This is the core function call for Step 1
    original_span_data, model_ready_features = processor.extract_features(NEW_PDF_PATH)
    
    if model_ready_features is not None:
        print("\n--- Features Ready for Model Prediction ---")
        print("Shape of the feature DataFrame:", model_ready_features.shape)
        print("\nFirst 5 rows of features:")
        print(model_ready_features.head())
        
        print("\n--- Corresponding Original Span Data ---")
        # Print the first 5 original spans to show the raw data
        for i in range(min(5, len(original_span_data))):
            print(original_span_data[i])

