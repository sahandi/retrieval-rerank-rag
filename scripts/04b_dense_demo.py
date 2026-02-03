from src.retrieval.dense import DenseRetriever

def main():
    r = DenseRetriever("indexes/chroma_moviedocs")
    query = "space adventure with aliens and a hero"
    hits = r.search(query, top_n=10)
    for doc, score in hits[:5]:
        print(score, doc["movieId"], doc["title"], "|", doc.get("genres"))

if __name__ == "__main__":
    main()
