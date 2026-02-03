# scripts/00_check_env.py
import os
import sys
import platform

def main():
    print("Python:", sys.version)
    print("Executable:", sys.executable)
    print("Platform:", platform.platform())

    print("\nHF cache env vars:")
    for k in ["HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"]:
        print(f"{k} =", os.environ.get(k))

    print("\nImport checks:")
    import pandas, numpy, datasets, chromadb, sentence_transformers, rank_bm25, sklearn  # noqa
    print("OK: imports succeeded")

if __name__ == "__main__":
    main()
