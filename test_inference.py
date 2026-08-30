import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from pytorch_model import MAMoEForCausalLM, MAMoEConfig

def generate_text(model, tokenizer, prompt, max_new_tokens=20):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
    print(f"\n🧠 Prompt: '{prompt}'")
    print("🤖 Output IDs: ", end="", flush=True)
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            
        print(f"{next_token} ", end="", flush=True)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
            
    print("\n")
    return tokenizer.decode(input_ids[0].numpy())

def main():
    config = MAMoEConfig()
    model = MAMoEForCausalLM(config)
    state_dict = torch.load("pytorch_model.pt", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    tokenizer = Tokenizer.from_file("tokenizer_hf/tokenizer.json")
    
    prompts = ["User: Ibukota Indonesia adalah?\nAssistant:"]
    for prompt in prompts:
        out = generate_text(model, tokenizer, prompt)
        print("Decoded text:", out)

if __name__ == "__main__":
    main()
