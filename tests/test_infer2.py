import torch
from tokenizers import Tokenizer
from pytorch_model import MAMoEForCausalLM, MAMoEConfig

tok_path = 'tokenizer_hf/tokenizer.json'
model_path = 'pytorch_model.pt'
tokenizer = Tokenizer.from_file(tok_path)
config = MAMoEConfig()
model = MAMoEForCausalLM(config)
state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
model.load_state_dict(state_dict, strict=False)
model.eval()

prompt = "User: Halo, perkenalkan saya Budi.\nAssistant:"
input_ids = tokenizer.encode(prompt).ids
input_tensor = torch.tensor([input_ids], dtype=torch.long)
with torch.no_grad():
    logits = model(input_tensor)
    next_token_logits = logits[0, -1, :]
    print("Top 5 token IDs:", torch.topk(next_token_logits, 5))
