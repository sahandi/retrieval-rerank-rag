
# Retrieval → Rerank → Generate (Multi-turn Movie Recommender)

A production-style conversational movie recommender built on **ReDial** using a **Retrieve → Rerank → Generate** pipeline:

1) **Retrieve (fast)**: BM25 + dense embeddings over MovieDocs  
2) **Rerank (smart)**: (Week 2) cross-encoder reranker trained with hard negatives  
3) **Generate (grounded)**: (Week 2) LLM response grounded in retrieved MovieDoc snippets  

This repo currently implements **Week 1**: MovieDocs + BM25 + dense + hybrid retrieval + evaluation harness.

---

## Quickstart

### 1) Clone
```bash
git clone https://github.com/sahandi/retrieval-rerank-rag.git
cd retrieval-rerank-rag
````

2) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -U pip
pip install datasets pandas numpy rank-bm25 sentence-transformers chromadb
```

### 4) (Optional) Put Hugging Face caches somewhere with free space

Hugging Face datasets/models can be large. You can redirect cache locations:

```bash
export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
```

---

## Project layout

```
scripts/              # runnable scripts
src/retrieval/        # retrieval modules (bm25, dense, hybrid)
data/
  raw/                # local datasets (NOT committed)
  processed/          # processed artifacts (NOT committed)
  sample/             # small samples safe to commit
indexes/              # chroma index (NOT committed)
outputs/              # small json result artifacts (committed)
```

---

## Data

### MovieLens 25M

Download **MovieLens 25M** from GroupLens and unzip to:

`data/raw/ml-25m/`

Must contain at least:

* `movies.csv`
* `links.csv`

### ReDial (Hugging Face)

Downloaded through the HF `datasets` loader (custom code requires `trust_remote_code=True`).

---

## Week 1: Build + run retrieval baselines

### 0) Sanity check environment

```bash
python scripts/00_check_env.py
```

### 1) Download / check datasets

```bash
python scripts/01_download_redial.py
python scripts/01b_check_movielens.py
```

### 2) Build MovieDocs (from MovieLens movies.csv)

```bash
python scripts/02_build_moviedocs.py
```

Output:

* `data/processed/moviedocs.jsonl` (NOT committed)
* `data/sample/moviedocs_sample.jsonl` (committed)

### 3) BM25 demo

Because code is organized under `src/`, run demo scripts using module mode:

```bash
python -m scripts.03_bm25_demo
```

### 4) Dense index (Chroma) + demo

```bash
python scripts/04_build_chroma_index.py
python -m scripts.04b_dense_demo
```

### 5) Hybrid retrieval demo

```bash
python -m scripts.05_hybrid_demo
```

---

## Evaluation 

We evaluate candidate generation with:

* **Recall@10**
* **NDCG@10**

### Important: ID-space mismatch + overlap-only evaluation

The `recwizard/redial` distribution contains movie identifiers that only **partially overlap** MovieLens IDs.
Because retrieval candidates come from **MovieLens**, we report metrics on the subset where the ground-truth label exists in MovieLens.

In our run:

* Total eval rows: 2000
* Overlap with MovieLens ID space: **729 (36.45%)**

### Run evaluation

```bash
python scripts/06_make_eval_dataset.py
python -m scripts.07_eval_retrieval
cat outputs/week1_eval_overlap_only.json
```

### Results (overlap-only, n = 729)

| System                            | Recall@10 | NDCG@10 |
| --------------------------------- | --------: | ------: |
| BM25                              |    0.0439 |  0.0300 |
| Dense (all-MiniLM-L6-v2 + Chroma) |    0.1166 |  0.0934 |
| Hybrid (naive merge)              |    0.0535 |  0.0434 |

Saved artifacts:

*`outputs/retrieval_eval.json`: retrieval metrics computed over all eval rows (includes ID-space mismatches).
* `outputs/retrieval_eval_overlap_only.json`: retrieval metrics computed only on examples whose ground-truth IDs exist in the MovieLens ID space (recommended for fair comparison).


---


