import os
import torch
import jax
import orbax.checkpoint as ocp
import numpy as np
from pytorch_model import MAMoEForCausalLM, MAMoEConfig

def convert_jax_to_pytorch(jax_params, pt_model, config):
    state_dict = {}
    
    def to_torch(x, transpose=False):
        x = np.array(x)
        if transpose:
            x = x.T
        return torch.from_numpy(x)

    # 1. Embeddings & Norm
    state_dict['embed_tokens.weight'] = to_torch(jax_params['embed_tokens']['embedding'])
    state_dict['embed_proj.weight'] = to_torch(jax_params['embed_proj']['kernel'], True)
    state_dict['embed_proj.bias'] = to_torch(jax_params['embed_proj']['bias'])
    state_dict['lm_head_proj.weight'] = to_torch(jax_params['lm_head_proj']['kernel'], True)
    state_dict['lm_head_proj.bias'] = to_torch(jax_params['lm_head_proj']['bias'])
    state_dict['norm.weight'] = to_torch(jax_params['norm']['weight'])
    
    # 2. Memory Controller
    state_dict['memory_controller.read_gate.weight'] = to_torch(jax_params['memory_controller']['read_gate']['kernel'], True)
    state_dict['memory_controller.read_gate.bias'] = to_torch(jax_params['memory_controller']['read_gate']['bias'])
    state_dict['memory_controller.write_gate.weight'] = to_torch(jax_params['memory_controller']['write_gate']['kernel'], True)
    state_dict['memory_controller.write_gate.bias'] = to_torch(jax_params['memory_controller']['write_gate']['bias'])
    
    # 2.5 Memory Bank
    state_dict['memory_bank.q_proj.weight'] = to_torch(jax_params['memory_bank']['q_proj']['kernel'], True)
    state_dict['memory_bank.k_proj.weight'] = to_torch(jax_params['memory_bank']['k_proj']['kernel'], True)
    state_dict['memory_bank.v_proj.weight'] = to_torch(jax_params['memory_bank']['v_proj']['kernel'], True)
    state_dict['memory_bank.i_proj.weight'] = to_torch(jax_params['memory_bank']['importance_proj']['kernel'], True)
    state_dict['memory_bank.i_proj.bias'] = to_torch(jax_params['memory_bank']['importance_proj']['bias'])
    state_dict['memory_bank.fusion_proj.weight'] = to_torch(jax_params['memory_bank']['fusion_proj']['kernel'], True)

    # 3. Layers
    for i in range(config.num_hidden_layers):
        layer_prefix = f'layers_{i}'
        pt_prefix = f'layers.{i}'
        
        # Norms
        state_dict[f'{pt_prefix}.input_layernorm.weight'] = to_torch(jax_params[layer_prefix]['input_layernorm']['weight'])
        state_dict[f'{pt_prefix}.post_attention_layernorm.weight'] = to_torch(jax_params[layer_prefix]['post_attention_layernorm']['weight'])
        
        # Attention
        attn = jax_params[layer_prefix]['self_attn']
        state_dict[f'{pt_prefix}.self_attn.q_proj.weight'] = to_torch(attn['q_proj']['kernel'], True)
        state_dict[f'{pt_prefix}.self_attn.k_proj.weight'] = to_torch(attn['k_proj']['kernel'], True)
        state_dict[f'{pt_prefix}.self_attn.v_proj.weight'] = to_torch(attn['v_proj']['kernel'], True)
        state_dict[f'{pt_prefix}.self_attn.o_proj.weight'] = to_torch(attn['o_proj']['kernel'], True)
        
        # MoE
        moe = jax_params[layer_prefix]['moe']
        state_dict[f'{pt_prefix}.moe.router.weight'] = to_torch(moe['MoERouter_0']['gate_proj']['kernel'], True)
        
        for j in range(config.num_experts):
            expert = moe[f'expert_{j}']
            state_dict[f'{pt_prefix}.moe.experts.{j}.gate_up_proj.weight'] = to_torch(expert['gate_up_proj']['kernel'], True)
            state_dict[f'{pt_prefix}.moe.experts.{j}.down_proj.weight'] = to_torch(expert['down_proj']['kernel'], True)

    # Load into PyTorch model to verify compatibility
    pt_model.load_state_dict(state_dict, strict=False)
    return state_dict

def main():
    ckpt_dir = '/kaggle/working/checkpoints/phase2'
    if not os.path.exists(ckpt_dir):
        ckpt_dir = '/kaggle/working/checkpoints/phase1'
        
    if not os.path.exists(ckpt_dir):
        print("❌ No JAX checkpoint found to export!")
        return

    print(f"Loading JAX checkpoint from {ckpt_dir}...")
    checkpointer = ocp.StandardCheckpointer()
    raw_state = checkpointer.restore(os.path.abspath(ckpt_dir), target=None)
    
    if 'params' in raw_state:
        jax_params = raw_state['params']
    else:
        jax_params = raw_state

    print("Initializing PyTorch model structure...")
    config = MAMoEConfig()
    pt_model = MAMoEForCausalLM(config)
    
    print("Converting and transposing weights...")
    state_dict = convert_jax_to_pytorch(jax_params, pt_model, config)
    
    output_path = '/kaggle/working/pytorch_model.pt'
    print(f"Saving PyTorch state_dict to {output_path}...")
    torch.save(state_dict, output_path)
    
    print("✅ Successfully exported to PyTorch format!")
    print("You can now download 'pytorch_model.pt' and load it using 'pytorch_model.py'!")

if __name__ == "__main__":
    main()
