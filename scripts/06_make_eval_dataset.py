# scripts/06_make_eval_dataset.py
from datasets import load_dataset
import json
import re
from pathlib import Path
import pandas as pd

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

MOVIES_CSV = Path("data/raw/ml-25m/movies.csv")
ID_RE = re.compile(r"@(\d+)")

def build_id2title():
    movies = pd.read_csv(MOVIES_CSV)  # movieId,title,genres
    # movieId is int, title is str
    return dict(zip(movies["movieId"].astype(int), movies["title"].astype(str)))

def expand_ids_to_titles(text: str, id2title: dict[int, str]) -> str:
    # Replace @12345 with "Title (Year)" when possible
    def repl(m):
        mid = int(m.group(1))
        title = id2title.get(mid)
        return title if title else f"@{mid}"
    return ID_RE.sub(repl, text)

def main():
    ds = load_dataset("recwizard/redial", trust_remote_code=True)
    split = "test" if "test" in ds else ("validation" if "validation" in ds else "train")

    id2title = build_id2title()

    examples = []
    max_rows = 2000
    tail_turns = 6

    for ex in ds[split]:
        messages = ex.get("messages")
        senders = ex.get("senders")
        movie_ids = ex.get("movieIds")

        if not messages or not senders or len(messages) != len(senders):
            continue
        if not movie_ids:
            continue

        tail = list(zip(senders, messages))[-tail_turns:]
        query_text = "\n".join([f"{s}: {m}" for s, m in tail if m])

        # KEY FIX: expand @12345 tokens into real titles
        query_text = expand_ids_to_titles(query_text, id2title)

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
    # quick sanity: show one expanded line
    print("Sample query:", examples[0]["query"][:200])

if __name__ == "__main__":
    main()
