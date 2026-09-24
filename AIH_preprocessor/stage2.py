import json
import os
from collections import Counter

# --- Configuration ---
# Directory where your 'complete' JSON files from Stage 1 are located.
INPUT_DIR = "labeled_data_complete" 
# Directory where the final, 'enriched' JSON files will be saved.
OUTPUT_DIR = "labeled_data_enriched"
# A tolerance for checking if a span is centered. A span is "centered" if its 
# middle point is within 5% of the page's width from the page's center.
CENTER_TOLERANCE_FACTOR = 0.05

def get_base_font_size(spans):
    """Calculates the most common font size for spans labeled as BODY_TEXT."""
    body_text_sizes = [
        s["font_size"] for s in spans if s.get("label_type") == "BODY_TEXT" and s.get("font_size") is not None
    ]
    if not body_text_sizes:
        # Fallback: use the most common font size in the whole document
        all_sizes = [s["font_size"] for s in spans if s.get("font_size") is not None]
        if not all_sizes:
            return 12.0 # A reasonable default if no font sizes are found
        return Counter(all_sizes).most_common(1)[0][0]
    
    return Counter(body_text_sizes).most_common(1)[0][0]

def preprocess_json_file(input_path, output_path):
    """
    Reads a 'complete' JSON file, adds normalized and contextual features, 
    and saves the final 'enriched' version.
    """
    print(f"Processing {input_path}...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            spans = json.load(f)
    except FileNotFoundError:
        print(f"  -> Error: File not found.")
        return
    except json.JSONDecodeError:
        print(f"  -> Error: Could not decode JSON from file.")
        return

    # --- Step 1: Add a temporary 'y_pos' from bbox for sorting ---
    for span in spans:
        if 'bbox' not in span:
            print(f"  -> Critical Error: Span with text '{span['text'][:30]}...' is missing 'bbox'. Aborting.")
            return
        # 'y_pos' is the top vertical coordinate of the span's bounding box
        span['y_pos'] = span['bbox'][1]

    # --- Step 2: Sort spans by reading order (page, then vertical position) ---
    spans_sorted = sorted(spans, key=lambda s: (s['page'], s['y_pos']))

    # --- Step 3: Pre-calculate document-wide features ---
    base_font_size = get_base_font_size(spans_sorted)
    print(f"  - Base font size for document: {base_font_size:.2f}")

    # --- Step 4: Iterate and add new features ---
    for i, current_span in enumerate(spans_sorted):
        # --- A. Normalized Features ---
        
        # 1. relative_font_size
        relative_font_size = current_span["font_size"] / base_font_size if base_font_size > 0 else 1.0
        
        # 2. is_centered
        page_width = current_span.get("page_width")
        is_centered = 0
        if page_width:
            bbox = current_span["bbox"]
            span_center_x = (bbox[0] + bbox[2]) / 2.0
            page_center_x = page_width / 2.0
            tolerance = page_width * CENTER_TOLERANCE_FACTOR
            if abs(span_center_x - page_center_x) < tolerance:
                is_centered = 1
        
        # --- B. Contextual Features ---
        
        # 3. distance_from_previous
        distance_from_previous = -1.0 # Default for the first span on a page or in the doc

        if i > 0:
            previous_span = spans_sorted[i-1]
            if current_span['page'] == previous_span['page']:
                # Vertical gap between the top of the current span and the bottom of the previous one.
                distance_from_previous = current_span['bbox'][1] - previous_span['bbox'][3]
        
        # --- Append the new features to the 9 base features ---
        # (labels saved by labeller.py already carry these 3 features, so
        # drop any existing ones to keep 12 features, matching prediction)
        current_span["features"] = current_span["features"][:9] + [
            relative_font_size,
            float(is_centered),
            distance_from_previous
        ]
        
    # --- Step 5: Clean up temporary and raw data, then save ---
    for span in spans_sorted:
        del span['y_pos']
        # These are no longer needed for the ML model, so we can remove them for cleanliness
        if 'bbox' in span: del span['bbox']
        if 'page_width' in span: del span['page_width']

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(spans_sorted, f, indent=4)
    
    print(f"  -> Saved enriched file to {output_path}")

def main():
    """Main function to find and process all JSON files in the input directory."""
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        print(f"Please create it and place your 9 'complete' JSON files from Stage 1 inside.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: '{OUTPUT_DIR}'")

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            input_file_path = os.path.join(INPUT_DIR, filename)
            # Save with the same name in the output directory
            output_file_path = os.path.join(OUTPUT_DIR, filename)
            preprocess_json_file(input_file_path, output_file_path)

if __name__ == "__main__":
    main()
