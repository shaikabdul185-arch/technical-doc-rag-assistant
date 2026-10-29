"""Hybrid RAG Engine: Semantic Chunking, BM25 + Dense SVD Retrieval, RRF, and Citation Synthesis."""
import math
import re
from collections import Counter
from typing import Dict, List
import numpy as np

STOPWORDS = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by", "is", "are", "what", "how", "why", "when"}


def tokenize_words(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_\-]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


class DocumentChunker:
    def __init__(self, strategy: str = "semantic_section") -> None:
        self.strategy = strategy

    def chunk_corpus(self, documents: List[Dict]) -> List[Dict]:
        chunks: List[Dict] = []
        for doc in documents:
            doc_id, title = doc["doc_id"], doc["title"]
            sections = [s.strip() for s in doc["content"].split("## ") if s.strip()]
            for idx, sec in enumerate(sections):
                lines = sec.split("\n", 1)
                header = lines[0].strip()
                body = lines[1].strip() if len(lines) > 1 else ""
                chunks.append({"chunk_id": f"{doc_id}#C{idx + 1}", "doc_id": doc_id, "title": title, "section": header, "text": f"{title} - {header}: {body}"})
        return chunks


class HybridRAGRetriever:
    def __init__(self, bm25_k1: float = 1.5, bm25_b: float = 0.75, embedding_dim: int = 6, rrf_k: int = 60) -> None:
        self.bm25_k1, self.bm25_b, self.embedding_dim, self.rrf_k = bm25_k1, bm25_b, embedding_dim, rrf_k
        self.chunks: List[Dict] = []
        self.doc_tokens: List[List[str]] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 1.0
        self.vocab: Dict[str, int] = {}
        self.proj: np.ndarray = np.array([])
        self.embs: np.ndarray = np.array([])

    def fit(self, chunks: List[Dict]) -> "HybridRAGRetriever":
        self.chunks = chunks
        self.doc_tokens = [tokenize_words(c["text"]) for c in chunks]
        n = len(chunks)
        self.avgdl = float(np.mean([len(t) for t in self.doc_tokens])) if n else 1.0
        dfs = Counter(term for toks in self.doc_tokens for term in set(toks))
        self.idf = {t: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for t, df in dfs.items()}
        self.vocab = {t: i for i, t in enumerate(sorted(self.idf.keys()))}
        tfidf = np.zeros((n, len(self.vocab)), dtype=float)
        for d_i, toks in enumerate(self.doc_tokens):
            for term, cnt in Counter(toks).items():
                tfidf[d_i, self.vocab[term]] = (1.0 + math.log(cnt)) * self.idf[term]
        _, s, vt = np.linalg.svd(tfidf, full_matrices=False)
        k = min(self.embedding_dim, len(s))
        self.proj = vt[:k, :].T
        raw = tfidf @ self.proj
        self.embs = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
        return self

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        q_toks = tokenize_words(query)
        bm25 = np.zeros(len(self.chunks), dtype=float)
        for i, toks in enumerate(self.doc_tokens):
            cnts = Counter(toks)
            dl = len(toks)
            bm25[i] = sum(self.idf.get(q, 0.0) * (cnts[q] * (self.bm25_k1 + 1.0)) / (cnts[q] + self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * (dl / self.avgdl))) for q in q_toks if q in cnts)
        q_vec = np.zeros(len(self.vocab), dtype=float)
        for t, c in Counter(q_toks).items():
            if t in self.vocab:
                q_vec[self.vocab[t]] = (1.0 + math.log(c)) * self.idf[t]
        q_emb = q_vec @ self.proj
        norm = float(np.linalg.norm(q_emb))
        dense = (self.embs @ (q_emb / norm)) if norm > 1e-12 else np.zeros(len(self.chunks))
        b_ranks = np.argsort(np.argsort(-bm25)) + 1
        d_ranks = np.argsort(np.argsort(-dense)) + 1
        scores = (1.0 / (self.rrf_k + b_ranks)) + (1.0 / (self.rrf_k + d_ranks)) + 0.01 * bm25
        idx = np.argsort(-scores)[:top_k]
        return [{**self.chunks[int(j)], "score": round(float(scores[int(j)]), 5), "bm25_score": round(float(bm25[int(j)]), 4)} for j in idx]


def synthesize_grounded_answer(query: str, hits: List[Dict], min_bm25: float = 1.0) -> Dict:
    if not hits or hits[0]["bm25_score"] < min_bm25:
        return {"query": query, "abstained": True, "answer": "Abstained: insufficient grounding confidence.", "citations": []}
    top = hits[0]
    return {"query": query, "abstained": False, "answer": f"{top['text']} [{top['chunk_id']}]", "citations": [top["chunk_id"]], "primary_doc_id": top["doc_id"]}
