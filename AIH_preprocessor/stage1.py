import fitz  # PyMuPDF
import json
import os
from collections import defaultdict

def create_lookup_from_labeled_data(labeled_spans):
    """
    Creates a lookup dictionary from the hand-labeled data.
    The key is a tuple: (page_number, stripped_text).
    The value is a list of span objects, to handle duplicate text on the same page.
    """
    lookup = defaultdict(list)
    for span in labeled_spans:
        # Use .strip() to remove leading/trailing whitespace, making matches more robust.
        key = (span['page'], span['text'].strip())
        lookup[key].append(span)
    return lookup

def amend_labeled_data(pdf_path, labeled_json_path, output_json_path):
    """
    Parses a PDF, matches spans against existing labeled data using text and page,
    and adds 'bbox' and 'page_width' to the labeled data.
    """
    print(f"Processing '{pdf_path}'...")

    # --- Step 1: Load the hand-labeled data and create a lookup table ---
    try:
        with open(labeled_json_path, 'r', encoding='utf-8') as f:
            labeled_spans = json.load(f)
    except FileNotFoundError:
        print(f"  -> ERROR: Labeled JSON file not found at '{labeled_json_path}'")
        return

    lookup_dict = create_lookup_from_labeled_data(labeled_spans)
    
    # We will populate this list with the amended data.
    # This list will be smaller if some labeled spans are not found in the new parse.
    amended_spans = []
    
    match_count = 0
    total_labeled = len(labeled_spans)

    # --- Step 2: Open the PDF and parse it with PyMuPDF ---
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  -> ERROR: Could not open or process PDF '{pdf_path}': {e}")
        return

    # --- Step 3: Iterate through new spans, match, and amend ---
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_width = page.rect.width
        # Using "dict" format is convenient as it gives font, bbox, text etc.
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    current_page_index = page_num + 1 # PyMuPDF is 0-indexed, your labels are 1-indexed
                    current_text = span["text"].strip()
                    
                    # Create the key to find a match in our lookup dictionary
                    lookup_key = (current_page_index, current_text)

                    if lookup_key in lookup_dict and lookup_dict[lookup_key]:
                        # Match found!
                        # Get the corresponding labeled span. We use pop(0) to correctly
                        # handle cases where the same text appears multiple times on one page.
                        labeled_span_to_amend = lookup_dict[lookup_key].pop(0)

                        # Add the new raw features
                        labeled_span_to_amend['bbox'] = list(span['bbox']) # Convert tuple to list for JSON
                        labeled_span_to_amend['page_width'] = page_width
                        
                        amended_spans.append(labeled_span_to_amend)
                        match_count += 1
                        
    print(f"  -> Matching complete. Found and amended {match_count} out of {total_labeled} labeled spans.")

    # Report on any labeled spans that were not found in the new parse
    unmatched_count = 0
    for key, remaining_spans in lookup_dict.items():
        if remaining_spans:
            unmatched_count += len(remaining_spans)
            print(f"  -> WARNING: Could not find labeled span on page {key[0]} with text: '{key[1][:50]}...'")
    
    if unmatched_count == 0 and match_count == total_labeled:
        print("  -> Success! All labeled spans were matched and amended.")

    # --- Step 4: Save the newly amended data to the output file ---
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(amended_spans, f, indent=4)
    print(f"  -> Saved amended data to '{output_json_path}'")


# --- HOW TO USE ---
if __name__ == "__main__":
    # 1. Define the paths for ONE pdf and its corresponding json
    PDF_PATH = "./AIH_model/AIH_preprocessor/source_pdfs/Set13_feature.pdf"
    LABELED_JSON_PATH = "./AIH_model/AIH_data/Set13_target.json"
    
    # 2. Define where to save the output
    OUTPUT_DIR = "labeled_data_complete"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Construct a descriptive output filename
    output_filename = os.path.basename(LABELED_JSON_PATH).replace('.json', '_complete.json')
    OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, output_filename)

    # 3. Run the function
    amend_labeled_data(PDF_PATH, LABELED_JSON_PATH, OUTPUT_JSON_PATH)

    # You would repeat this for all 9 of your PDF/JSON pairs.
