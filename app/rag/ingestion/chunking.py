"""Chunking rules for FAQ source material."""


def chunk_faqs(text: str) -> list[str]:
    """Split FAQ content where every question begins with ``Q:``.

    Text before the first Q: is ignored: an FAQ vector should contain a question
    and its associated answer, not document headings or boilerplate.
    """
    chunks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("Q:"):
            if current:
                chunk = "\n".join(current).strip()
                if chunk:
                    chunks.append(chunk)
            current = [line]
        elif current:
            current.append(line)

    if current:
        chunk = "\n".join(current).strip()
        if chunk:
            chunks.append(chunk)
    return chunks
