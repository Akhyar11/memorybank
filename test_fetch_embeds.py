import torch
from transformers import AutoTokenizer, AutoModel
import sys

model_id = "indolem/indobert-base-uncased"
print(f"Loading {model_id}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Vocab size: {tokenizer.vocab_size}")
    model = AutoModel.from_pretrained(model_id)
    embeds = model.embeddings.word_embeddings.weight.data
    print(f"Embeddings shape: {embeds.shape}")
except Exception as e:
    print(f"Error: {e}")
