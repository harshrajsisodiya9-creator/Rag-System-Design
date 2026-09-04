"""Retrieve FAQ context from FAISS and generate an answer with ChatGroq."""

from __future__ import annotations

import asyncio
import os

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.rag.ingestion.service import hybrid_search

RETRIEVAL_TOP_K = 4


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer FAQ questions using only the supplied context. "
            "If the context does not contain the answer, say that you do not know. "
            "Do not invent policies, facts, or links.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {query}"),
    ]
)


def format_context(documents: list[Document]) -> str:
    """Convert retrieved FAQ chunks into clearly separated prompt context."""
    return "\n\n---\n\n".join(document.page_content for document in documents)


def get_chat_model() -> ChatGroq:
    """Create the Groq chat client from the process environment."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not configured")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
    )


async def answer_query(query: str) -> str:
    """Run retrieval, context construction, and Groq answer generation."""
    documents = await hybrid_search(query, k=RETRIEVAL_TOP_K)
    if not documents:
        return "I do not know based on the available FAQ context."

    messages = PROMPT.format_messages(context=format_context(documents), query=query)
    model = get_chat_model()
    response = await model.ainvoke(messages)
    return str(response.content)
