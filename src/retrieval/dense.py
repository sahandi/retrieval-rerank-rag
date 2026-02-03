# src/retrieval/dense.py
from typing import List, Dict, Tuple
import chromadb
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, chroma_path: str, collection_name: str = "moviedocs"):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.col = self.client.get_or_create_collection(collection_name)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def search(self, query: str, top_n: int = 10) -> List[Tuple[Dict, float]]:
        q_emb = self.model.encode([query]).tolist()[0]
        res = self.col.query(query_embeddings=[q_emb], n_results=top_n, include=["metadatas", "distances"])
        out = []
        for mid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
            # Chroma returns "distance" (smaller = closer). Convert to similarity-ish score.
            score = float(-dist)
            out.append(({"movieId": int(mid), **meta}, score))
        return out
