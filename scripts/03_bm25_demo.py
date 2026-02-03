# scripts/03_bm25_demo.py
from src.retrieval.bm25 import BM25Retriever

def main():
    r = BM25Retriever("data/processed/moviedocs.jsonl").build()
    query = "romantic comedy with wedding"
    hits = r.search(query, top_n=10)
    for doc, score in hits[:5]:
        print(score, doc["movieId"], doc["title"], "|", doc["genres"])

if __name__ == "__main__":
    main()
