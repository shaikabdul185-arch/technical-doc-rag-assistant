"""FastAPI service for Hybrid RAG Technical Documentation Assistant."""
from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_engine import DocumentChunker, HybridRAGRetriever, synthesize_grounded_answer

DOCS = [
    {"doc_id": "DOC-001", "title": "PyTorch DDP", "content": "## Gradient Bucketing\nDDP overlaps backward pass with NCCL ring-AllReduce using bucket_cap_mb=25."},
    {"doc_id": "DOC-002", "title": "Model Drift PSI", "content": "## Alerting Thresholds\nPSI >= 0.25 triggers automated Airflow retraining DAGs."},
    {"doc_id": "DOC-003", "title": "LLM PagedAttention", "content": "## KV Cache\nPagedAttention partitions KV cache into non-contiguous blocks to eliminate memory fragmentation."},
]

CHUNKS = DocumentChunker().chunk_corpus(DOCS)
RETRIEVER = HybridRAGRetriever().fit(CHUNKS)
app = FastAPI(title="Hybrid RAG Technical Documentation Assistant API")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 2


@app.get("/health")
def health_check():
    return {"status": "healthy", "indexed_chunks": len(CHUNKS)}


@app.post("/query")
def query_docs(req: QueryRequest):
    hits = RETRIEVER.search(req.query, top_k=req.top_k)
    return synthesize_grounded_answer(req.query, hits)
