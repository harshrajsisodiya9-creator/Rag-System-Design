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

The response reports each uploaded file independently. Parsing/chunking errors do not stop other files. Embedding calls use exponential-backoff retries (up to four attempts); an exhausted batch failure is reported for only files represented in that batch, and subsequent batches continue. The FAISS index is process-local and intentionally not persisted because retrieval is not included yet.
