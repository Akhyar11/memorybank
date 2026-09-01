import os
import json
from dataclasses import dataclass

@dataclass
class TokenizerSpec:
    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    unk_token_id: int

    @classmethod
    def from_file(cls, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer config not found at {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # We need to extract vocabulary size and special token IDs.
        # tokenizers library JSON usually has a "model" -> "vocab" dict, or we can find it in "added_tokens".
        # If it's a HuggingFace tokenizer.json, the format is standard.
        
        vocab_size = 0
        if "model" in data and "vocab" in data["model"]:
            vocab_size = len(data["model"]["vocab"])
            
        pad_token_id = None
        bos_token_id = None
        eos_token_id = None
        unk_token_id = None
        
        # Usually added_tokens is a list of dicts with 'id' and 'content'
        if "added_tokens" in data:
            for token_obj in data["added_tokens"]:
                content = token_obj.get("content", "")
                token_id = token_obj.get("id")
                
                # Identify special tokens based on standard names
                if content in ["[PAD]", "<pad>"]:
                    pad_token_id = token_id
                elif content in ["[CLS]", "<s>", "<|im_start|>"]:
                    bos_token_id = token_id
                elif content in ["[SEP]", "</s>", "<|im_end|>"]:
                    eos_token_id = token_id
                elif content in ["[UNK]", "<unk>"]:
                    unk_token_id = token_id
                    
        # Fallback to defaults or specific token IDs if not explicitly marked, 
        # but in our strict requirement we should ideally error if not found.
        # We will try to map common names from the vocab if added_tokens doesn't have them.
        if "model" in data and "vocab" in data["model"]:
            vocab = data["model"]["vocab"]
            if pad_token_id is None: pad_token_id = vocab.get("[PAD]", vocab.get("<pad>"))
            if bos_token_id is None: bos_token_id = vocab.get("[CLS]", vocab.get("<s>", vocab.get("<|im_start|>")))
            if eos_token_id is None: eos_token_id = vocab.get("[SEP]", vocab.get("</s>", vocab.get("<|im_end|>")))
            if unk_token_id is None: unk_token_id = vocab.get("[UNK]", vocab.get("<unk>"))
            
        if pad_token_id is None: raise ValueError("pad_token_id not found in tokenizer.json")
        if bos_token_id is None: raise ValueError("bos_token_id not found in tokenizer.json")
        if eos_token_id is None: raise ValueError("eos_token_id not found in tokenizer.json")
        if unk_token_id is None: raise ValueError("unk_token_id not found in tokenizer.json")

        return cls(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id
        )
