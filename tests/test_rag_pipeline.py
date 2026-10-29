"""Unit and integration tests for Hybrid RAG Technical Documentation Assistant."""
import unittest
from app.main import QueryRequest, health_check, query_docs


class TestHybridRAG(unittest.TestCase):
    def test_health(self):
        self.assertEqual(health_check()["status"], "healthy")

    def test_grounded_retrieval_and_citation(self):
        res = query_docs(QueryRequest(query="How does PagedAttention eliminate KV cache memory fragmentation?", top_k=2))
        self.assertFalse(res["abstained"])
        self.assertEqual(res["primary_doc_id"], "DOC-003")
        self.assertIn("DOC-003#C1", res["citations"])

    def test_out_of_domain_abstention(self):
        res = query_docs(QueryRequest(query="How do I bake chocolate sourdough bread?", top_k=2))
        self.assertTrue(res["abstained"])


if __name__ == "__main__":
    unittest.main()
