# src/retrieval/bm25.py
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]

class BM25Retriever:
    def __init__(self, moviedocs_path: str):
        self.moviedocs_path = Path(moviedocs_path)
        self.docs: List[Dict] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25 = None

    def build(self):
        self.docs = []
        self.corpus_tokens = []
        with self.moviedocs_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.docs.append(obj)
                self.corpus_tokens.append(tokenize(obj["text"]))
        self.bm25 = BM25Okapi(self.corpus_tokens)
        return self

    def search(self, query: str, top_n: int = 10) -> List[Tuple[Dict, float]]:
        assert self.bm25 is not None, "Call build() first"
        q_tok = tokenize(query)
        scores = self.bm25.get_scores(q_tok)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [(self.docs[i], float(scores[i])) for i in top_idx]
