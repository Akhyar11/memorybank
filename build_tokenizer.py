import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm

def main():
    vocab_size = 32000
    out_dir = "tokenizer"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tokenizer.json")

    if os.path.exists(out_path):
        print(f"Tokenizer already exists at {out_path}")
        return

    print("Loading VQFat Indonesian Corpus from local file...")
    dataset = load_dataset("csv", data_files="data/raw/vqfat_cosmopedia_id.csv", split="train")
    
    # Auto-detect text column
    text_column = "text"
    if "text" not in dataset.column_names:
        for col in ["prompt", "content", "completion", "text_clean", "article"]:
            if col in dataset.column_names:
                text_column = col
                break
    print(f"Using column '{text_column}' for text data.")
    
    # Use a small subset (e.g. 5%) for vocabulary training to save time
    dataset = dataset.select(range(int(len(dataset) * 0.05)))
    
    print(f"Loaded {len(dataset)} articles. Training BPE Tokenizer (vocab size: {vocab_size})...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    special_tokens = ["[UNK]", "[PAD]", "<EOS>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2
    )
    
    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size][text_column]
            
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    tokenizer.post_processor = TemplateProcessing(
        single="$A <EOS>",
        special_tokens=[
            ("<EOS>", tokenizer.token_to_id("<EOS>"))
        ]
    )
    
    tokenizer.save(out_path)
    print(f"Tokenizer successfully trained and saved to {out_path} 🚀")

if __name__ == "__main__":
    main()
