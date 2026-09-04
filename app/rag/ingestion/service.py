"""Fault-tolerant FAQ ingestion service backed by LangChain FAISS."""

from __future__ import annotations

import asyncio
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import UploadFile

load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rank_bm25 import BM25Okapi
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .chunking import chunk_faqs
from .readers import extract_text

EMBEDDING_BATCH_SIZE = 50
MAX_EMBEDDING_ATTEMPTS = 4


@dataclass
class FileReport:
    filename: str
    status: str
    chunks: int = 0
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "filename": self.filename,
            "status": self.status,
            "chunks": self.chunks,
        }
        if self.reason:
            result["reason"] = self.reason
        return result


_index_lock = asyncio.Lock()
_vector_store: FAISS | None = None
_indexed_documents: list[Document] = []
_bm25_index: BM25Okapi | None = None


def tokenize(text: str) -> list[str]:
    """Use the same lightweight tokenization for indexed text and queries."""
    return re.findall(r"\w+", text.lower())


def rebuild_bm25_index() -> None:
    """Rebuild the lexical index from chunks successfully added to FAISS."""
    global _bm25_index
    _bm25_index = BM25Okapi(
        [tokenize(document.page_content) for document in _indexed_documents]
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    )


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(MAX_EMBEDDING_ATTEMPTS),
    reraise=True,
)
def embed_with_backoff(
    embeddings: GoogleGenerativeAIEmbeddings, texts: list[str]
) -> list[list[float]]:
    """Embed one batch, retrying API failures with exponential backoff."""
    return embeddings.embed_documents(texts)


def batches(items: list[Document], size: int) -> list[list[Document]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


# search_index() itself does not modify FAISS. The lock is there to prevent a lookup
# from running at the same time as /ingest is adding vectors.
async def search_index(query: str, k: int = 4) -> list[Document]:
    """Retrieve the nearest FAQ chunks from the in-memory FAISS index."""
    async with _index_lock:
        if _vector_store is None:
            raise LookupError("No FAQ chunks have been ingested yet")
        return await asyncio.to_thread(_vector_store.similarity_search, query, k=k)


async def hybrid_search(
    query: str, k: int = 4, rrf_constant: int = 60
) -> list[Document]:
    """Fuse dense FAISS and lexical BM25 results with reciprocal-rank fusion."""
    async with _index_lock:
        if _vector_store is None:
            raise LookupError("No FAQ chunks have been ingested yet")

        candidate_count = min(len(_indexed_documents), max(k * 3, k))
        dense_documents = await asyncio.to_thread(
            _vector_store.similarity_search, query, k=candidate_count
        )
        lexical_documents: list[Document] = []
        query_tokens = tokenize(query)
        if _bm25_index is not None and query_tokens:
            scores = _bm25_index.get_scores(query_tokens)
            ranked_indexes = sorted(
                range(len(scores)), key=lambda index: scores[index], reverse=True
            )
            lexical_documents = [
                _indexed_documents[index] for index in ranked_indexes[:candidate_count]
            ]

        fused_scores: defaultdict[str, float] = defaultdict(float)
        documents_by_id: dict[str, Document] = {}
        for ranked_documents in (dense_documents, lexical_documents):
            for rank, document in enumerate(ranked_documents, start=1):
                document_id = str(document.metadata["document_id"])
                documents_by_id[document_id] = document
                fused_scores[document_id] += 1 / (rrf_constant + rank)

        ranked_ids = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)
        return [documents_by_id[document_id] for document_id in ranked_ids[:k]]


async def ingest_files(files: list[UploadFile]) -> dict[str, Any]:
    """Ingest uploads, reporting each file independently after failures."""
    reports: dict[str, FileReport] = {}
    documents: list[Document] = []
    document_report_keys: dict[str, str] = {}

    for ordinal, upload in enumerate(files):
        filename = upload.filename or f"upload-{ordinal + 1}"
        report_key = f"{ordinal}:{filename}"
        try:
            text = await extract_text(upload)
            faq_chunks = chunk_faqs(text)
            if not faq_chunks:
                raise ValueError(
                    "No FAQ chunks found; questions must start exactly with 'Q:'"
                )
            reports[report_key] = FileReport(
                filename=filename, status="pending", chunks=len(faq_chunks)
            )
            for chunk_number, chunk in enumerate(faq_chunks):
                document_id = str(uuid4())
                document_report_keys[document_id] = report_key
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "chunk": chunk_number,
                            "document_id": document_id,
                        },
                    )
                )
        except Exception as exc:
            reports[report_key] = FileReport(
                filename=filename, status="failed", reason=str(exc)
            )

    if documents:
        try:
            embeddings = get_embeddings()
        except Exception as exc:
            for report in reports.values():
                if report.status == "pending":
                    report.status = "failed"
                    report.reason = str(exc)
        else:
            async with _index_lock:
                global _vector_store
                for batch in batches(documents, EMBEDDING_BATCH_SIZE):
                    try:
                        vectors = await asyncio.to_thread(
                            embed_with_backoff,
                            embeddings,
                            [document.page_content for document in batch],
                        )
                        pairs = list(
                            zip(
                                [document.page_content for document in batch],
                                vectors,
                                strict=True,
                            )
                        )
                        metadatas = [document.metadata for document in batch]
                        if _vector_store is None:
                            _vector_store = FAISS.from_embeddings(
                                pairs, embeddings, metadatas=metadatas
                            )
                        else:
                            _vector_store.add_embeddings(pairs, metadatas=metadatas)
                        _indexed_documents.extend(batch)
                        rebuild_bm25_index()
                    except Exception as exc:
                        for document in batch:
                            report = reports[
                                document_report_keys[document.metadata["document_id"]]
                            ]
                            if report.status != "failed":
                                report.status = "failed"
                                report.reason = f"Embedding batch failed: {exc}"

            for report in reports.values():
                if report.status == "pending":
                    report.status = "success"

    values = [report.as_dict() for report in reports.values()]
    return {
        "files": values,
        "successful_files": sum(report["status"] == "success" for report in values),
        "failed_files": sum(report["status"] == "failed" for report in values),
    }
