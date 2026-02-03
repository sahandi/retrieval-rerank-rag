# src/retrieval/hybrid.py
from typing import Dict, List, Tuple
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever

class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever):
        self.bm25 = bm25
        self.dense = dense

    def search(self, query: str, top_bm25: int = 50, top_dense: int = 50, top_k: int = 50) -> List[Dict]:
        bm_hits = self.bm25.search(query, top_n=top_bm25)
        de_hits = self.dense.search(query, top_n=top_dense)

        merged: Dict[int, Dict] = {}

        for doc, score in bm_hits:
            mid = int(doc["movieId"])
            merged[mid] = {
                "movieId": mid,
                "title": doc["title"],
                "genres": doc["genres"],
                "bm25_score": float(score),
                "dense_score": None,
            }

        for doc, score in de_hits:
            mid = int(doc["movieId"])
            if mid not in merged:
                merged[mid] = {
                    "movieId": mid,
                    "title": doc.get("title"),
                    "genres": doc.get("genres"),
                    "bm25_score": None,
                    "dense_score": float(score),
                }
            else:
                merged[mid]["dense_score"] = float(score)

        # Simple hybrid ranking rule for week1:
        # normalize by presence: prefer items that appear in both lists.
        def rank_key(x):
            both = (x["bm25_score"] is not None) and (x["dense_score"] is not None)
            return (1 if both else 0,
                    (x["bm25_score"] or 0.0),
                    (x["dense_score"] or -1e9))

        ranked = sorted(merged.values(), key=rank_key, reverse=True)
        return ranked[:top_k]
