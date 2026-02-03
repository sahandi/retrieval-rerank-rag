# scripts/05_hybrid_demo.py
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever

def main():
    bm25 = BM25Retriever("data/processed/moviedocs.jsonl").build()
    dense = DenseRetriever("indexes/chroma_moviedocs")
    hy = HybridRetriever(bm25, dense)

    query = "a dark psychological thriller about memory"
    hits = hy.search(query, top_bm25=50, top_dense=50, top_k=10)
    for h in hits[:5]:
        print(h["movieId"], h["title"], "| bm25:", h["bm25_score"], "dense:", h["dense_score"])

if __name__ == "__main__":
    main()
