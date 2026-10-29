# Hybrid RAG Technical Documentation Assistant & IR Evaluation Engine

An end-to-end Retrieval-Augmented Generation (RAG) and Information Retrieval (IR) engineering project combining sparse lexical (**Okapi BM25**) and dense semantic (**SVD/LSA Vector**) search via **Reciprocal Rank Fusion (RRF)**, inline `[DOC-ID#CHUNK-ID]` citations, and out-of-domain abstention guardrails.

## 1. Retrieval Evaluation Benchmark (N = 20 Technical Queries)

| Retrieval Mode | Hit Rate@1 | Hit Rate@3 | MRR@5 | NDCG@5 | Mean Query Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense Semantic (SVD/LSA)** | 0.8500 | 1.0000 | 0.9083 | 0.9315 | 0.254 ms |
| **Sparse Lexical (BM25)** | 0.9500 | 1.0000 | 0.9750 | 0.9815 | 0.189 ms |
| **Hybrid RRF (Production)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.452 ms** |

## 2. Chunking Strategy Ablation Study

| Chunking Strategy | Indexed Chunks | Hit Rate@1 | Hit Rate@3 | MRR@5 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fixed Window (35 words)** | 25 | 0.9500 | 1.0000 | 0.9750 | 0.9815 |
| **Sliding Overlap (45w, 15w overlap)** | 25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Semantic Section-Aware** | **24** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

## 3. Quickstart

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -v
uvicorn app.main:app --reload --port 8000
```
