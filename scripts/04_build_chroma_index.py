# scripts/04_build_chroma_index.py
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

MOVIEDOCS = Path("data/processed/moviedocs.jsonl")
CHROMA_DIR = Path("indexes/chroma_moviedocs")

def main():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # :contentReference[oaicite:26]{index=26}
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))  # :contentReference[oaicite:27]{index=27}
    col = client.get_or_create_collection("moviedocs")

    ids, texts, metadatas = [], [], []

    with MOVIEDOCS.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(str(obj["movieId"]))
            texts.append(obj["text"])
            metadatas.append({"title": obj["title"], "genres": obj["genres"]})

    # Batch embed (safe size)
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        t_batch = texts[i:i+batch_size]
        id_batch = ids[i:i+batch_size]
        m_batch = metadatas[i:i+batch_size]
        emb = model.encode(t_batch, show_progress_bar=False).tolist()
        col.upsert(ids=id_batch, documents=t_batch, embeddings=emb, metadatas=m_batch)

    print("Indexed docs:", col.count(), "at", CHROMA_DIR)

if __name__ == "__main__":
    main()
