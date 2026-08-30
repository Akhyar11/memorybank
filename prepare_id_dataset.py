import argparse
import os
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm

def train_tokenizer(dataset, vocab_size, output_path, text_column="text"):
    print(f"Training BPE Tokenizer (vocab size: {vocab_size})...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    special_tokens = ["[UNK]", "[PAD]", "<EOS>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
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
    
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    return tokenizer

def tokenize_and_chunk(dataset, tokenizer, seq_len, text_column="text"):
    print(f"Tokenizing dataset and chunking into seq_len={seq_len}...")
    
    def tokenize_function(examples):
        encoded = tokenizer.encode_batch(examples[text_column])
        return {"tokens": [enc.ids for enc in encoded]}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=4
    )
    
    print("Flattening tokens...")
    all_tokens = []
    for item in tqdm(tokenized_dataset):
        all_tokens.extend(item["tokens"])
        
    total_tokens = len(all_tokens)
    total_chunks = total_tokens // seq_len
    print(f"Total tokens: {total_tokens:,}. Generated {total_chunks:,} chunks of length {seq_len}.")
    
    all_tokens = all_tokens[:total_chunks * seq_len]
    
    # vocab_size 32k fits cleanly inside uint16
    chunks = np.array(all_tokens, dtype=np.uint16).reshape((total_chunks, seq_len))
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Prepare Indonesian Pretraining Dataset for JAX")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--out-dir", type=str, default="data/pretrain", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading VQFat Indonesian Corpus from local file...")
    dataset = load_dataset("csv", data_files="data/raw/vqfat_cosmopedia_id.csv", split="train")
    
    print(f"Loaded {len(dataset):,} articles.")
    
    # Auto-detect text column
    text_column = "text"
    if "text" not in dataset.column_names:
        for col in ["prompt", "content", "completion", "text_clean", "article"]:
            if col in dataset.column_names:
                text_column = col
                break
    print(f"Using column '{text_column}' for text data.")
    
    tokenizer_path = os.path.join(args.out_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        tokenizer = train_tokenizer(dataset, args.vocab_size, tokenizer_path, text_column)
    else:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        
    chunks = tokenize_and_chunk(dataset, tokenizer, args.seq_len, text_column)
    
    out_file = os.path.join(args.out_dir, "train_chunks.npy")
    print(f"Saving numpy array to {out_file}...")
    np.save(out_file, chunks)
    print("Done! Data is ready for JAX dataloading.")

if __name__ == "__main__":
    main()
