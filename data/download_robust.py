#!/usr/bin/env python3
"""Robust data downloader with retry logic."""
import os
import time
from huggingface_hub import hf_hub_download

LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fineweb10B")
REPO_ID = "kjj0/fineweb10B-gpt2"
MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds


def download_file(fname):
    path = os.path.join(LOCAL_DIR, fname)
    if os.path.exists(path):
        return True
    for attempt in range(MAX_RETRIES):
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
            )
            print(f"  Downloaded: {fname}")
            return True
        except Exception as e:
            print(f"  Attempt {attempt+1}/{MAX_RETRIES} failed for {fname}: {type(e).__name__}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return False


if __name__ == "__main__":
    os.makedirs(LOCAL_DIR, exist_ok=True)
    files = ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, 51)]
    total = len(files)
    for i, fname in enumerate(files):
        exists = os.path.exists(os.path.join(LOCAL_DIR, fname))
        status = "exists" if exists else "downloading"
        print(f"[{i+1}/{total}] {fname} ({status})")
        if not exists:
            if not download_file(fname):
                print(f"FAILED: {fname} after {MAX_RETRIES} retries")
    existing = len([f for f in files if os.path.exists(os.path.join(LOCAL_DIR, f))])
    print(f"\nDone: {existing}/{total} files ready")
