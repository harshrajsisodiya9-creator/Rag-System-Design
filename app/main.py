"""HTTP entry point for the FAQ RAG service."""

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.rag.ingestion.service import ingest_files

app = FastAPI(title="FAQ RAG ingestion API", version="0.1.0")


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)) -> dict:
    """Accept multiple FAQ documents and pass them to the RAG ingestion layer."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    return await ingest_files(files)
