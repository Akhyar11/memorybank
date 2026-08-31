import os
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from pytorch_model import MAMoEForCausalLM, MAMoEConfig

def load_model_and_tokenizer(model_path='pytorch_model.pt', tok_path='tokenizer_hf/tokenizer.json'):
    print("Memuat tokenizer...")
    if not os.path.exists(tok_path):
        print(f"❌ Tokenizer tidak ditemukan di {tok_path}")
        sys.exit(1)
    tokenizer = Tokenizer.from_file(tok_path)

    print("Memuat arsitektur model...")
    config = MAMoEConfig()
    model = MAMoEForCausalLM(config)
    
    print(f"Memuat bobot model dari {model_path}...")
    if not os.path.exists(model_path):
        print(f"❌ Model PyTorch tidak ditemukan di {model_path}.")
        print("Silakan jalankan 'export_pytorch.py' di Kaggle dan unduh file 'pytorch_model.pt' ke folder ini.")
        sys.exit(1)
        
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    # Deteksi perangkat (GPU jika ada, jika tidak pakai CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan perangkat: {device}")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

import time

def generate(model, tokenizer, device, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    # Dapatkan ID untuk token pemisah/akhir (tergantung dataset, biasanya [SEP] atau <|im_end|>)
    eos_token_id = tokenizer.token_to_id("[SEP]") 
    if eos_token_id is None:
        eos_token_id = tokenizer.token_to_id("<|im_end|>")
        
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    generated = input_ids.copy()
    
    print("\nAI: ", end="", flush=True)
    
    start_time = time.time()
    num_generated = 0
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :]
            
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
            
            num_generated += 1
            
            # Jika memprediksi End-of-Sequence, berhenti
            if next_token == eos_token_id:
                break
                
            generated.append(next_token)
            input_tensor = torch.tensor([generated], dtype=torch.long).to(device)
            
            # Decode token terbaru saja dan print secara streaming
            # Catatan: WordPiece kadang memunculkan tanda '##', kita hapus untuk tampilan rapi
            new_word = tokenizer.decode([next_token])
            if new_word.startswith("##"):
                new_word = new_word[2:]
            elif len(generated) > len(input_ids) + 1:
                new_word = " " + new_word
                
            print(new_word, end="", flush=True)
            
    end_time = time.time()
    elapsed = end_time - start_time
    tok_per_sec = num_generated / elapsed if elapsed > 0 else 0
    
    print(f"\n\n[⏱️ Kecepatan: {tok_per_sec:.2f} tok/detik | Total: {num_generated} token]")

def main():
    print("🚀 Inisialisasi Memory Bank - Local Chat Mode\n")
    model, tokenizer, device = load_model_and_tokenizer()
    
    print("\nSiap! Ketik 'keluar' atau 'exit' untuk berhenti.")
    print("-" * 50)
    
    # Riwayat percakapan sederhana
    chat_history = ""
    
    while True:
        try:
            user_input = input("\nAnda: ")
            if user_input.strip().lower() in ['keluar', 'exit', 'quit']:
                break
            if user_input.strip() == "":
                continue
                
            # Bentuk format prompt percakapan (Sesuaikan dengan format di t5gemma2)
            # Biasanya bentuknya seperti: "User: Halo\nAssistant:"
            prompt = f"{chat_history}User: {user_input}\nAssistant:"
            
            generate(model, tokenizer, device, prompt, max_new_tokens=150, temperature=0.7)
            
            # Simpan sejarah singkat agar ada konteks multi-turn
            # (Kita batasi agar tidak OOM di GPU MX350)
            chat_history = prompt # Simpan konteks terakhir saja (atau bisa diakumulasi jika mau)
            
        except KeyboardInterrupt:
            print("\nSampai jumpa!")
            break

if __name__ == "__main__":
    main()
