import fitz
import pytest

BODY = [
    "Plain body text that explains the section in a few words",
    "and continues on a second line of ordinary paragraph text",
    "before finishing on a third line of the same paragraph.",
]

# (page, text, font size, bold) -- the outline a correct extractor should find
HEADINGS = [
    (1, "Annual Research Report", 24, True),   # title
    (1, "1 Introduction", 18, True),           # H1
    (1, "1.1 Background", 15, True),           # H2
    (1, "1.1.1 Scope", 13, True),              # H3
    (2, "2 Methods", 18, True),                # H1
    (2, "2.1 Data Sources", 15, True),         # H2
]
SIZE_TO_LABEL = {24: 1, 18: 2, 15: 3, 13: 4, 11: 0}  # TITLE, H1, H2, H3, BODY


def write_sample_pdf(path):
    """Two-page PDF with a centred title, H1-H3 headings and body paragraphs."""
    doc = fitz.open()
    for page_no in (1, 2):
        page = doc.new_page(width=595, height=842)
        y = 80
        for heading_page, text, size, bold in HEADINGS:
            if heading_page != page_no:
                continue
            font = "hebo" if bold else "helv"
            x = 72
            if size == 24:  # centre the title
                x = (page.rect.width - fitz.get_text_length(text, fontname=font, fontsize=size)) / 2
            page.insert_text((x, y), text, fontsize=size, fontname=font)
            y += size + 24
            for line in BODY:
                page.insert_text((72, y), line, fontsize=11, fontname="helv")
                y += 14
            y += 30
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def sample_pdf(tmp_path):
    return write_sample_pdf(tmp_path / "sample.pdf")
