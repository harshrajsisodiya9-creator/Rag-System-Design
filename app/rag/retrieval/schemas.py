"""Request and response schemas for RAG queries."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="Question to answer from the FAQ index")


class QueryResponse(BaseModel):
    response: str
