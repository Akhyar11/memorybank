import torch
state_dict = torch.load("pytorch_model.pt", map_location='cpu', weights_only=True)
print("Embed weight norm:", state_dict['embed_tokens.weight'].norm().item())
print("Layer 0 Attention norm:", state_dict['layers.0.self_attn.q_proj.weight'].norm().item())
