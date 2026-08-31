import torch
from tokenizers import Tokenizer
from pytorch_model import MAMoEForCausalLM, MAMoEConfig
from chat import generate

def main():
    print("Loading model for test...")
    tok_path = 'tokenizer_hf/tokenizer.json'
    model_path = 'pytorch_model.pt'
    
    tokenizer = Tokenizer.from_file(tok_path)
    config = MAMoEConfig()
    model = MAMoEForCausalLM(config)
    
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    prompt = "User: Halo, perkenalkan saya Budi.\nAssistant:"
    print(f"Prompt: {prompt}")
    generate(model, tokenizer, device, prompt, max_new_tokens=50, temperature=0.7)

if __name__ == "__main__":
    main()
