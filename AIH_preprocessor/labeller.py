import fitz  # PyMuPDF
import json
import os
from collections import Counter, defaultdict

# --- Configuration ---
# The final output file where all labeled and enriched data will be stored.
OUTPUT_JSON_FILE = "training_data_enriched.json"
# A tolerance for checking if a span is centered.
CENTER_TOLERANCE_FACTOR = 0.05

class PDFLabeler:
    def __init__(self, output_file):
        self.output_file = output_file
        self.all_labeled_data = self._load_existing_data()
        # Keep track of texts already labeled to avoid re-labeling
        self.existing_keys = {(item['page'], item['text']) for item in self.all_labeled_data}
        self.current_pdf_spans = []
        
        self.label_types = {
            "0": "BODY_TEXT", "1": "TITLE", "2": "H1",
            "3": "H2", "4": "H3", "5": "H4",
            "s": "SKIP", "q": "QUIT"
        }
        self.label_prompt = "Label [ (0)Body (1)Title (2)H1 (3)H2 (4)H3 (5)H4 | (s)Skip (q)Quit & Save ]: "

    def _load_existing_data(self):
        """Loads existing labeled data from the output file."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✅ Loaded {len(data)} existing labels from {self.output_file}")
                    return data
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"⚠️ Could not load or parse {self.output_file}, starting fresh.")
                return []
        print(f"📝 No existing data file found. Starting fresh.")
        return []

    def load_pdf(self, pdf_path):
        """Extracts all text spans from a PDF for labeling."""
        if not os.path.exists(pdf_path):
            print(f"❌ Error: PDF file not found at {pdf_path}")
            return False

        print(f"\n📄 Loading PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        self.current_pdf_spans = []

        for page_num, page in enumerate(doc):
            page_width = page.rect.width
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block: continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        # Skip tiny spans or already labeled spans from previous sessions
                        if len(text) < 3 or (page_num + 1, text) in self.existing_keys:
                            continue

                        self.current_pdf_spans.append({
                            "text": text,
                            "page": page_num + 1,
                            "font_size": span["size"],
                            "is_bold": "Bold" in span["font"],
                            "is_italic": "Italic" in span["font"],
                            "x_pos": span["bbox"][0],
                            "bbox": span["bbox"],
                            "page_width": page_width
                        })
        
        doc.close()
        print(f"🔍 Found {len(self.current_pdf_spans)} new spans to label.")
        return True

    def start_labeling_session(self):
        """Starts the interactive labeling process for the loaded PDF."""
        newly_labeled_spans = []
        total_spans = len(self.current_pdf_spans)

        for i, span in enumerate(self.current_pdf_spans):
            print("-" * 50)
            print(f"Processing span {i+1}/{total_spans} | Page: {span['page']}")
            print(f"Text: \"{span['text']}\"")
            
            label_input = input(self.label_prompt).lower()

            if label_input == 'q':
                print("🛑 Quitting and saving progress...")
                break
            if label_input == 's':
                print("⏭️ Skipping span.")
                continue
            if label_input not in self.label_types:
                print("⚠️ Invalid input. Skipping span.")
                continue

            # --- Create the initial feature vector ---
            features = [
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

            labeled_item = {
                "text": span["text"],
                "features": features,
                "label": int(label_input),
                "label_type": self.label_types[label_input],
                "page": span["page"],
                "font_size": span["font_size"],
                "is_bold": span["is_bold"],
                # --- Carry over the raw data needed for enrichment ---
                "bbox": span["bbox"],
                "page_width": span["page_width"]
            }
            newly_labeled_spans.append(labeled_item)
        
        if newly_labeled_spans:
            self._finalize_and_save(newly_labeled_spans)

    def _finalize_and_save(self, new_spans):
        """Enriches the new spans with advanced features and saves all data."""
        print("\n✨ Finalizing and enriching new labels...")
        
        # --- Step 1: Enrich the newly labeled spans ---
        enriched_spans = self._enrich_spans_with_features(new_spans)
        
        # --- Step 2: Add enriched spans to the main data list ---
        self.all_labeled_data.extend(enriched_spans)
        
        # --- Step 3: Save the combined data to the JSON file ---
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.all_labeled_data, f, indent=4)
            print(f"💾 Successfully saved {len(self.all_labeled_data)} total labeled items to {self.output_file}")
            # Update the set of existing keys for the next PDF
            self.existing_keys.update({(item['page'], item['text']) for item in enriched_spans})
        except Exception as e:
            print(f"🔥 Critical Error: Could not save data! - {e}")

    def _enrich_spans_with_features(self, spans_to_enrich):
        """Calculates and adds the normalized and contextual features."""
        # Sort by reading order, which is essential for contextual features
        spans_sorted = sorted(spans_to_enrich, key=lambda s: (s['page'], s['bbox'][1]))
        
        # Calculate base font size for this document's BODY_TEXT
        body_text_sizes = [s['font_size'] for s in spans_sorted if s['label_type'] == 'BODY_TEXT']
        if not body_text_sizes:
            all_sizes = [s['font_size'] for s in spans_sorted]
            base_font_size = Counter(all_sizes).most_common(1)[0][0] if all_sizes else 12.0
        else:
            base_font_size = Counter(body_text_sizes).most_common(1)[0][0]

        print(f"  - Calculated base font size for this batch: {base_font_size:.2f}pt")

        for i, span in enumerate(spans_sorted):
            # --- Normalized Features ---
            relative_font_size = span["font_size"] / base_font_size if base_font_size > 0 else 1.0
            
            span_center_x = (span["bbox"][0] + span["bbox"][2]) / 2.0
            page_center_x = span["page_width"] / 2.0
            tolerance = span["page_width"] * CENTER_TOLERANCE_FACTOR
            is_centered = 1.0 if abs(span_center_x - page_center_x) < tolerance else 0.0
            
            # --- Contextual Features ---
            distance_from_previous = -1.0 # Default value
            if i > 0:
                prev_span = spans_sorted[i-1]
                if span['page'] == prev_span['page']:
                    distance_from_previous = span['bbox'][1] - prev_span['bbox'][3]
            
            # Append the new features to the existing list
            span['features'].extend([
                relative_font_size,
                is_centered,
                distance_from_previous
            ])

        # Clean up raw data from the final JSON object to keep it clean
        # (after the loop: the next span still needs the previous span's bbox)
        for span in spans_sorted:
            del span['bbox']
            del span['page_width']

        return spans_sorted

# --- How to Run the Script ---
if __name__ == "__main__":
    labeler = PDFLabeler(OUTPUT_JSON_FILE)
    
    # Example: Process a PDF from a folder named 'source_pdfs'
    pdf_directory = "source_pdfs" # <--- PUT YOUR PDFs IN THIS FOLDER
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        print(f"Created directory '{pdf_directory}'. Please place your PDFs there.")
    
    pdf_files_to_process = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    
    if not pdf_files_to_process:
        print(f"No PDFs found in '{pdf_directory}'. Please add some PDFs to continue.")
    else:
        for pdf_file in pdf_files_to_process:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            if labeler.load_pdf(pdf_path):
                labeler.start_labeling_session()
        
        print("\n🎉 All PDFs have been processed.")
