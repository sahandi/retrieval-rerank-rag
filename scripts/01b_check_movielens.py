# scripts/01b_check_movielens.py
from pathlib import Path
import pandas as pd

def main():
    base = Path("data/raw/ml-25m")
    movies = pd.read_csv(base / "movies.csv")
    links = pd.read_csv(base / "links.csv")
    print("movies:", movies.shape, "columns:", list(movies.columns))
    print("links:", links.shape, "columns:", list(links.columns))
    print(movies.head(3))
    print(links.head(3))

if __name__ == "__main__":
    main()
