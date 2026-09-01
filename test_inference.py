import sys
import os
import time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts', 'pytorch'))
from pytorch_model import MAMoEForConditionalGeneration, MAMoEConfig

def generate(model, tokenizer, device, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, mem_state=None):
    model.eval()
    
    eos_token_id = tokenizer.token_to_id("[SEP]") 
    if eos_token_id is None:
        eos_token_id = tokenizer.token_to_id("<|im_end|>")
        
    bos_token_id = tokenizer.token_to_id("[CLS]")
    if bos_token_id is None:
        bos_token_id = tokenizer.token_to_id("<|im_start|>")
    if bos_token_id is None:
        bos_token_id = 1 # Fallback to 1 if unknown
        
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    decoder_input_ids = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)
    
    print(f"\nPrompt: {prompt}\nAI: ", end="", flush=True)
    start_time = time.time()
    num_generated = 0
    
    with torch.no_grad():
        # Precompute Encoder Pass to speed up generation
        bsz, enc_seq_len = input_tensor.shape
        x_enc = model.embed_proj(model.embed_tokens(input_tensor))
        cos_enc, sin_enc = model.rope(enc_seq_len)
        cos_enc = cos_enc.view(1, 1, enc_seq_len, -1)
        sin_enc = sin_enc.view(1, 1, enc_seq_len, -1)
        encoder_hidden_states = model.encoder(x_enc, cos_enc, sin_enc)
        h_prompt_eos = encoder_hidden_states[:, -1, :]
        read_prob, write_prob = model.memory_controller(h_prompt_eos)
        
        if mem_state is not None:
            memory_output, mem_state = model.memory_bank.read(h_prompt_eos, mem_state)
            fused_memory_context = (memory_output * read_prob).unsqueeze(1)
            full_context_states = torch.cat([fused_memory_context, encoder_hidden_states], dim=1)
        else:
            full_context_states = encoder_hidden_states
            
        encoder_outputs = (full_context_states, write_prob, mem_state)

        past_key_values = None
        generated_tokens = []
        printed_len = 0
        
        for i in range(max_new_tokens):
            is_eos = (i == max_new_tokens - 1)
            
            # Forward pass using KV Cache
            curr_decoder_input_ids = decoder_input_ids[:, -1:] if past_key_values is not None else decoder_input_ids
            
            out = model(input_tensor, curr_decoder_input_ids, mem_state=mem_state, is_eos=is_eos, 
                        encoder_outputs=encoder_outputs, past_key_values=past_key_values, use_cache=True)
            if mem_state is not None:
                logits, mem_state, write_prob, past_key_values = out
            else:
                logits, past_key_values = out
                write_prob = None
                
            next_token_logits = logits[0, -1, :]
            
            # CEGAH MODEL MEMPREDIKSI TOKEN 0 (PAD)
            next_token_logits[0] = float('-inf')
            
            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits).item()
            else:
                next_token_logits = next_token_logits / temperature
                
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
                out = model(input_tensor, decoder_input_ids, mem_state=mem_state, is_eos=True)
                if mem_state is not None:
                    _, mem_state, write_prob = out
                break
                
            generated_tokens.append(next_token)
            full_str = tokenizer.decode(generated_tokens)
            new_str = full_str[printed_len:]
            print(new_str, end="", flush=True)
            printed_len = len(full_str)
            
            decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            num_generated += 1
            
    end_time = time.time()
    elapsed = end_time - start_time
    tok_per_sec = num_generated / elapsed if elapsed > 0 else 0
    print(f"\n\n[⏱️ Kecepatan: {tok_per_sec:.2f} tok/detik | Total: {num_generated} token]\n")
    return mem_state

def main():
    print("Loading tokenizer & PyTorch Model...")
    tok_path = 'tokenizer_hf/tokenizer.json'
    model_path = '/home/akhyar/Unduhan/pytorch_model_epoch_20.pt'
    
    tokenizer = Tokenizer.from_file(tok_path)
    config = MAMoEConfig()
    model = MAMoEForConditionalGeneration(config)
    
    print(f"Loading state dict from {model_path}...")
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    if hasattr(model, 'stack_experts'):
        model.stack_experts()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan perangkat: {device}")
    model.to(device)
    
    mem_state = model.memory_bank.init_state(bsz=1, device=device)
    
    prompt = "User: Halo, bagaimana kabarmu hari ini?\nAssistant:"
    mem_state = generate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        mem_state=mem_state
    )

if __name__ == "__main__":
    main()
