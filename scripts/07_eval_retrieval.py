# scripts/07_eval_retrieval.py
import json
import math
from pathlib import Path
import csv

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


def recall_at_k(ranked_ids, label_id, k=10) -> float:
    return 1.0 if label_id in ranked_ids[:k] else 0.0


def ndcg_at_k(ranked_ids, label_id, k=10) -> float:
    for i, mid in enumerate(ranked_ids[:k]):
        if mid == label_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


def load_eval(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_movielens_ids(movies_csv="data/raw/ml-25m/movies.csv"):
    ids = set()
    with open(movies_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ids.add(int(row["movieId"]))
            except Exception:
                continue
    return ids


def main():
    eval_rows = load_eval("data/processed/eval_queries.jsonl")
    if not eval_rows:
        raise RuntimeError("eval_queries.jsonl is empty. Run scripts/06_make_eval_dataset.py first.")
    if "label_redial_movieId" not in eval_rows[0]:
        raise RuntimeError("Expected 'label_redial_movieId' in eval rows.")

    ml_ids = load_movielens_ids()

    # ✅ Filter rows to only those whose label exists in MovieLens (overlap)
    filtered = []
    for r in eval_rows:
        lab = int(r["label_redial_movieId"])
        if lab in ml_ids:
            filtered.append(r)

    print(f"Eval rows total: {len(eval_rows)}")
    print(f"Eval rows in MovieLens ID space: {len(filtered)} ({len(filtered)/max(len(eval_rows),1):.2%})")

    if not filtered:
        raise RuntimeError("No eval rows overlap with MovieLens IDs; cannot evaluate retrieval.")

    bm25 = BM25Retriever("data/processed/moviedocs.jsonl").build()
    dense = DenseRetriever("indexes/chroma_moviedocs")
    hybrid = HybridRetriever(bm25, dense)

    systems = {
        "bm25": lambda q: [int(d["movieId"]) for d, _s in bm25.search(q, top_n=10)],
        "dense": lambda q: [int(d["movieId"]) for d, _s in dense.search(q, top_n=10)],
        "hybrid": lambda q: [int(d["movieId"]) for d in hybrid.search(q, top_bm25=50, top_dense=50, top_k=10)],
    }

    results = {}
    for name, fn in systems.items():
        r_sum, n_sum, n = 0.0, 0.0, 0
        for row in filtered:
            q = row["query"]
            label_id = int(row["label_redial_movieId"])

            ranked_ids = fn(q)
            r_sum += recall_at_k(ranked_ids, label_id, k=10)
            n_sum += ndcg_at_k(ranked_ids, label_id, k=10)
            n += 1

        results[name] = {
            "n": n,
            "recall@10": r_sum / max(n, 1),
            "ndcg@10": n_sum / max(n, 1),
        }

    out = Path("outputs/week1_eval_overlap_only.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
