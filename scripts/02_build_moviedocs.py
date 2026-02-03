# scripts/02_build_moviedocs.py
from pathlib import Path
import json
import pandas as pd

RAW = Path("data/raw/ml-25m")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def clean_title(title: str) -> str:
    # MovieLens titles often contain year in parentheses at end
    return title.strip()

def main():
    movies = pd.read_csv(RAW / "movies.csv")  # columns: movieId,title,genres
    out_path = OUT / "moviedocs.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for row in movies.itertuples(index=False):
            movie_id = int(row.movieId)
            title = clean_title(row.title)
            genres = row.genres if isinstance(row.genres, str) else ""
            text = f"Title: {title}\nGenres: {genres}\n"

            obj = {
                "movieId": movie_id,
                "title": title,
                "genres": genres,
                "text": text,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Wrote:", out_path)

    # write a tiny sample for GitHub
    sample_path = Path("data/sample/moviedocs_sample.jsonl")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("r", encoding="utf-8") as fin, sample_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= 50:
                break
            fout.write(line)

    print("Wrote sample:", sample_path)

if __name__ == "__main__":
    main()
