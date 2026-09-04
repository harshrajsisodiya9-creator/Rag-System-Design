Practicing different approaches for a RAG based pipelines
, current progress includes chunking of either txt or pdf documents using structure aware chunking
which break chunks when stumbling upon Q: and includes answers with the chunk too

Example '.txt' or '.pdf' input:

```text
Q: How do I reset my password?
A: Select "Forgot password" from the sign-in page.

Q: How long does delivery take?
A: Delivery normally takes 3–5 business days.
```

The response reports each uploaded file independently. Parsing/chunking errors do not stop other files. Embedding calls use exponential-backoff retries (up to four attempts); an exhausted batch failure is reported for only files represented in that batch, and subsequent batches continue. The FAISS index is process-local and intentionally not persisted.

Both the ingestion and chunk retrieval includes locking of vector_store while either of them are performing
operations, this maintains atomic operations when multiple requests hits the server when ingestion only 
the first ingestion operation performs vector embedding into the FAISS in memory store this ensures atomic
insertion with latency trade-off, same for retrieval since we dont want retireval when any ingestion is taking place(error when either the ingestion is happening and vector_store is empty or current lookup
 is getting ingested),same with multiple reads(side effect of locking)

This is huge tradeoff when handling large number of requests, current implementation is just for research/learning purpose

Locking is required since we are performing in-memory vector_store operations this is not required when we are using a vector_store which either uses local or api calling since they have there own implementation of locking/transactions

## Hybrid retrieval

Each successfully indexed FAQ chunk is added to both FAISS (semantic/dense search) and an in-memory BM25 index (keyword/lexical search). At query time, both searches retrieve candidates and reciprocal-rank fusion combines their rankings before the top chunks become LLM context. This helps exact terms such as product names, error codes, and IDs while retaining semantic matching for differently phrased questions.
