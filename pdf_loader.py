from pathlib import Path
from typing import List
import PyPDF2


def pdf_to_text(path: Path) -> str:
    """Merge all pages into a single string."""
    with path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join(page.extract_text() or "" for page in reader.pages)