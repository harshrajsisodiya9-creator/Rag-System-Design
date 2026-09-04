"""HTTP entry point for the FAQ RAG service."""

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.rag.ingestion.service import ingest_files
from app.rag.retrieval.schemas import QueryRequest, QueryResponse
from app.rag.retrieval.service import answer_query

app = FastAPI(title="FAQ RAG ingestion API", version="0.1.0")


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)) -> dict:
    """Accept multiple FAQ documents and pass them to the RAG ingestion layer."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    return await ingest_files(files)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a question using chunks retrieved from the in-memory FAQ index."""
    try:
        return QueryResponse(response=await answer_query(request.query))
    except LookupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
