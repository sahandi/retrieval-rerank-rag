# scripts/01_download_export_redial_to_raw.py
from pathlib import Path
import shutil
from datasets import load_dataset

def main():
    out_dir = Path("data/raw/redial_hf")

    if out_dir.exists():
        print(f"Removing existing folder: {out_dir}")
        shutil.rmtree(out_dir)

    ds = load_dataset("recwizard/redial", trust_remote_code=True)
    ds.save_to_disk(str(out_dir))
    print("Saved DatasetDict to:", out_dir.resolve())

if __name__ == "__main__":
    main()