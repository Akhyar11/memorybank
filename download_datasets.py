import os
# Aktifkan HF_TRANSFER untuk download secepat kilat (menggunakan Rust)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from datasets import load_dataset

def main():
    out_dir = "data/raw"
    os.makedirs(out_dir, exist_ok=True)

    print("1. Downloading VQFat Indonesian Corpus (cosmopedia_id)...")
    try:
        vqfat = load_dataset("vickyfatrian/vqfat-indo-corpus", "cosmopedia_id", split="train")
        vqfat_path = os.path.join(out_dir, "vqfat_cosmopedia_id.parquet")
        vqfat.to_parquet(vqfat_path)
        print(f"Saved VQFat to {vqfat_path}")
    except Exception as e:
        print(f"Failed to download VQFat: {e}")

    print("\n2. Downloading T5Gemma2 Indonesia Chat Formatted...")
    configs = ['chat_multiturn', 'chat_sft']
    for config in configs:
        try:
            print(f"Downloading config: {config}")
            ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", config, split="train")
            path = os.path.join(out_dir, f"t5gemma2_{config}.parquet")
            ds.to_parquet(path)
            print(f"Saved {config} to {path}")
        except Exception as e:
            print(f"Failed to download {config}: {e}")

    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()
