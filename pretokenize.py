"""
pretokenize.py
==============
Jalankan SEKALI sebelum training untuk mengkonversi CSV mentah
menjadi numpy binary (.npy) yang bisa di-load super cepat saat training.

Output: /kaggle/working/vqfat_tokens.npy  (dtype=uint16, shape=[N])
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

# ─── Konfigurasi Path ────────────────────────────────────────────────────────
KAGGLE_CSV     = '/kaggle/input/datasets/akhyarsafrudin/dataset-chat/vqfat_cosmopedia_id.csv'
KAGGLE_TOK     = 'tokenizer_hf/tokenizer.json'
LOCAL_CSV      = 'data/raw/vqfat_cosmopedia_id.csv'
LOCAL_TOK      = 'tokenizer_hf/tokenizer.json'
OUTPUT_PATH    = '/kaggle/working/vqfat_tokens.npy'
OUTPUT_PATH_LO = 'data/pretrain/vqfat_tokens.npy'  # fallback lokal
COLAB_CSV      = '/content/drive/MyDrive/Colab Notebooks/dataset/vqfat_cosmopedia_id.csv'
COLAB_OUT      = '/content/drive/MyDrive/Colab Notebooks/dataset/vqfat_tokens.npy'
# ─────────────────────────────────────────────────────────────────────────────

def resolve_paths():
    if os.path.exists(KAGGLE_CSV):
        return KAGGLE_CSV, KAGGLE_TOK, OUTPUT_PATH
    elif os.path.exists(LOCAL_CSV):
        os.makedirs('data/pretrain', exist_ok=True)
        return LOCAL_CSV, LOCAL_TOK, OUTPUT_PATH_LO
    elif os.path.exists(COLAB_CSV):
        return COLAB_CSV, LOCAL_TOK, COLAB_OUT
    else:
        print("❌ Dataset CSV tidak ditemukan!")
        sys.exit(1)

def main():
    csv_path, tok_path, out_path = resolve_paths()

    if os.path.exists(out_path):
        arr = np.load(out_path, mmap_mode='r')
        print(f"✅ Pre-tokenized file sudah ada: {out_path}  ({arr.shape[0]:,} tokens)")
        return

    print(f"📄 Sumber CSV   : {csv_path}")
    print(f"🔤 Tokenizer    : {tok_path}")
    print(f"💾 Output        : {out_path}")
    print()

    tokenizer = Tokenizer.from_file(tok_path)

    # Deteksi kolom teks
    peek = pd.read_csv(csv_path, nrows=1)
    text_col = next(
        (c for c in ["text","prompt","content","completion","text_clean","article"]
         if c in peek.columns),
        peek.columns[0]
    )
    print(f"📌 Kolom teks: '{text_col}'")

    all_tokens = []
    t0 = time.time()
    total_rows = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=50000)):
        texts  = chunk[text_col].dropna().astype(str).tolist()
        total_rows += len(texts)
        encoded = tokenizer.encode_batch(texts)
        for enc in encoded:
            all_tokens.extend(enc.ids)

        elapsed = time.time() - t0
        print(f"  chunk {chunk_idx+1:4d} | rows so far: {total_rows:>8,} | tokens: {len(all_tokens):>12,} | {elapsed:.1f}s")

    print(f"\nTotal tokens: {len(all_tokens):,}")
    arr = np.array(all_tokens, dtype=np.uint16)
    np.save(out_path, arr)
    print(f"✅ Tersimpan di: {out_path}  ({arr.nbytes / 1e6:.1f} MB)")

if __name__ == '__main__':
    main()
