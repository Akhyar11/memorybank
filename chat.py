import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from pytorch_model import MAMoEForCausalLM, MAMoEConfig

import time

def generate(model, tokenizer, device, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, mem_state=None):
    model.eval()
    
    eos_token_id = tokenizer.token_to_id("[SEP]") 
    if eos_token_id is None:
        eos_token_id = tokenizer.token_to_id("<|im_end|>")
        
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    print("\nAI: ", end="", flush=True)
    start_time = time.time()
    num_generated = 0
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Write memory ONLY on the very last token generated in this turn
            is_eos = (i == max_new_tokens - 1)
            
            # Forward pass
            logits, mem_state, write_prob = model(input_tensor, mem_state=mem_state, is_eos=is_eos)
            next_token_logits = logits[0, -1, :]
            
            # CEGAH MODEL MEMPREDIKSI TOKEN 0 (PAD)
            next_token_logits[0] = float('-inf')
            
            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits).item()
            else:
                next_token_logits = next_token_logits / temperature
                
                # Top-p sampling (Nucleus)
                if top_p < 1.0:
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
                # Force memory write before exiting early
                _, mem_state, write_prob = model(input_tensor, mem_state=mem_state, is_eos=True)
                break
                
            token_str = tokenizer.decode([next_token])
            print(token_str, end=" ", flush=True)
            
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)
            num_generated += 1
            
            # Print Memory Write Event if prob > 0.5
            if write_prob is not None and write_prob[0].item() > 0.5:
                print(f"\n\033[92m[💾 AI menyimpan interaksi ini ke Memory Bank!]\033[0m")
                
    end_time = time.time()
    elapsed = end_time - start_time
    tok_per_sec = num_generated / elapsed if elapsed > 0 else 0
    print(f"\n\n[⏱️ Kecepatan: {tok_per_sec:.2f} tok/detik | Total: {num_generated} token]\n")
    return mem_state

def main():
    print("Loading tokenizer & PyTorch Model...")
    tok_path = 'tokenizer_hf/tokenizer.json'
    model_path = 'pytorch_model.pt'
    
    tokenizer = Tokenizer.from_file(tok_path)
    config = MAMoEConfig()
    model = MAMoEForCausalLM(config)
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} tidak ditemukan. Harap jalankan export_pytorch.py di Kaggle dan unduh hasilnya.")
        return
        
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan perangkat: {device}")
    model.to(device)
    
    print("\n✅ Siap! Ketik 'quit' atau 'exit' untuk keluar.\n")
    
    # Initialize Memory State!
    mem_state = model.memory_bank.init_state(bsz=1, device=device)
    
    while True:
        try:
            user_input = input("Anda: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            prompt = f"User: {user_input}\nAssistant:"
            
            # PENTING: Pass mem_state and receive updated mem_state
            mem_state = generate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                mem_state=mem_state
            )
            
        except KeyboardInterrupt:
            print("\nSampai jumpa!")
            break

if __name__ == "__main__":
    main()
