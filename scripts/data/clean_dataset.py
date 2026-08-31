import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import re
import argparse
import pandas as pd
from datasets import load_dataset, Dataset

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Hapus spasi di awal
    text = text.lstrip()
    # 2. Hapus yang bukan abjad, angka, atau simbol tertentu (, . - + = < > * /)
    text = re.sub(r'[^a-zA-Z0-9\s,\.\-\+\=\<\>\*\/]', '', text)
    return text

def main():
    parser = argparse.ArgumentParser(description="Pembersih Dataset Teks")
    parser.add_argument("--input", type=str, default="data/raw/wikipedia_id.parquet", help="Path file input (.parquet atau .csv)")
    parser.add_argument("--output", type=str, default="data/raw/wikipedia_id_clean.parquet", help="Path file output (.parquet atau .csv)")
    args = parser.parse_args()

    print(f"Loading dataset from {args.input}...")
    try:
        if args.input.endswith(".parquet"):
            df = pd.read_parquet(args.input)
        elif args.input.endswith(".csv"):
            df = pd.read_csv(args.input)
        else:
            print("Format file tidak didukung! Gunakan .parquet atau .csv")
            return
    except Exception as e:
        print(f"Gagal memuat dataset: {e}")
        return

    # Deteksi kolom teks
    col = next((c for c in ["text", "prompt", "content", "completion", "text_clean", "article"] if c in df.columns), None)
    if col is None:
        print("Kolom teks tidak ditemukan!")
        return

    print(f"Memulai pembersihan teks pada kolom '{col}' (Total baris: {len(df):,})...")
    df[col] = df[col].apply(clean_text)
    
    # Hapus baris yang menjadi kosong setelah dibersihkan
    df = df[df[col].str.strip() != ""]
    
    print(f"Menyimpan hasil ke {args.output} (Sisa baris: {len(df):,})...")
    if args.output.endswith(".parquet"):
        df.to_parquet(args.output, index=False)
    elif args.output.endswith(".csv"):
        df.to_csv(args.output, index=False)
        
    print("✅ Pembersihan dataset selesai!")

if __name__ == "__main__":
    main()
