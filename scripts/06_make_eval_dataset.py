# scripts/06_make_eval_dataset.py
from datasets import load_dataset
import json
from pathlib import Path

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    # Use trust_remote_code=True because recwizard/redial uses a custom loader
    ds = load_dataset("recwizard/redial", trust_remote_code=True)

    split = "test" if "test" in ds else ("validation" if "validation" in ds else "train")

    examples = []
    max_rows = 2000          # cap for speed
    tail_turns = 6           # last N turns to build query context

    for ex in ds[split]:
        # Your schema: lists of strings + list of ints
        messages = ex.get("messages")   # list[str]
        senders = ex.get("senders")     # list[str]
        movie_ids = ex.get("movieIds")  # list[int]

        # Basic validation
        if not messages or not senders or len(messages) != len(senders):
            continue
        if not movie_ids:
            continue

        # Build a query from the last few turns
        tail = list(zip(senders, messages))[-tail_turns:]
        query_text = "\n".join([f"{s}: {m}" for s, m in tail if m])

        # Create one eval row per labeled movieId
        for mid in movie_ids:
            examples.append({
                "query": query_text,
                "label_redial_movieId": int(mid),
            })
            if len(examples) >= max_rows:
                break

        if len(examples) >= max_rows:
            break

    out_path = OUT / "eval_queries.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print("Split:", split)
    print("Wrote:", out_path)
    print("Rows:", len(examples))

if __name__ == "__main__":
    main()
