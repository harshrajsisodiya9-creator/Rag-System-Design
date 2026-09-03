"""Input readers for supported FAQ upload formats."""

from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".txt", ".pdf"}


async def extract_text(upload: UploadFile) -> str:
    """Read a supported upload and return its textual content."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Only .txt and .pdf files are supported")

    content = await upload.read()
    if not content:
        raise ValueError("The uploaded file is empty")

    if suffix == ".txt":
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Text files must be UTF-8 encoded") from exc

    try:
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc
