import os
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import sys

sys.path.append('/kaggle/working/memorybank')
from pytorch_model import MAMoEForConditionalGeneration, MAMoEConfig

def test_tinystories():
    model_path = '/kaggle/working/pytorch_model.pt'
    tok_path = '/kaggle/working/memorybank/tokenizer_hf/tokenizer.json'
    
    if not os.path.exists(model_path):
        # Fallback to local
        model_path = 'pytorch_model.pt'
        tok_path = 'tokenizer_hf/tokenizer.json'
        
    print("⏳ Memuat Tokenizer...")
    tokenizer = Tokenizer.from_file(tok_path)
    
    print("⏳ Menginisialisasi Model PyTorch MAMoE-Tiny...")
    config = MAMoEConfig()
    model = MAMoEForConditionalGeneration(config)
    
    print(f"⏳ Memuat state_dict dari {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    except:
        print("⚠️ Model belum ditraining atau path salah, mencoba dengan bobot acak.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Memindahkan model ke: {device}")
    model.to(device)
    model.eval()
    
    # === SILAKAN UBAH PROMPT DI SINI ===
    prompt = "Pada suatu hari, Budi pergi ke taman. Di sana, Budi menemukan sebuah apel yang"
    # ===================================
    
    print(f"\n👤 Prompt: {prompt}")
    print("🤖 AI: ", end="", flush=True)
    
    # 1. ENCODER INPUT
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # 2. DECODER INPUT (Pancing menggunakan token terakhir dari prompt)
    last_token = input_ids[-1] if len(input_ids) > 0 else 0
    decoder_input = torch.tensor([[last_token]], dtype=torch.long).to(device)
    
    pad_token_id = tokenizer.token_to_id("<|pad|>") or 0
    eos_token_id = tokenizer.token_to_id("[SEP]") or tokenizer.token_to_id("<|im_end|>") or 0
        
    # 3. MEMORY BANK INIT
    mem_state = model.memory_bank.init_state(bsz=1, device=device)
    
    max_new_tokens = 50
    temperature = 0.7
    top_p = 0.9
    num_generated = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            is_eos = (i == max_new_tokens - 1)
            
            logits, mem_state, write_prob = model(input_tensor, decoder_input, mem_state=mem_state, is_eos=is_eos)
            
            next_token_logits = logits[0, -1, :]
            next_token_logits[pad_token_id] = float('-inf')
            
            # Top-p Sampling
            next_token_logits = next_token_logits / temperature
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == eos_token_id:
                _, mem_state, _ = model(input_tensor, decoder_input, mem_state=mem_state, is_eos=True)
                break
                
            token_str = tokenizer.decode([next_token])
            print(token_str, end=" ", flush=True)
            
            decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]], device=device)], dim=1)
            num_generated += 1

    elapsed = time.time() - start_time
    speed = num_generated / elapsed if elapsed > 0 else 0
    print(f"\n\n[⏱️ Selesai | {num_generated} token | {speed:.2f} tok/s]")

if __name__ == "__main__":
    test_tinystories()
