import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

def main():
    model_id = "indolem/indobert-base-uncased"
    print(f"Loading {model_id} from HuggingFace...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained("tokenizer_hf")
    print(f"Tokenizer saved to 'tokenizer_hf' with vocab size {tokenizer.vocab_size}")
    
    model = AutoModel.from_pretrained(model_id)
    embeds = model.embeddings.word_embeddings.weight.detach().numpy()
    
    print(f"Extracted embeddings shape: {embeds.shape}")
    
    np.save("pretrained_embeds.npy", embeds)
    print("Saved embeddings to 'pretrained_embeds.npy'")

if __name__ == "__main__":
    main()
