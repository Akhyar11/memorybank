import os
import sys
import torch
import jax
import jax.numpy as jnp
import numpy as np

# Setup paths to ensure we can import mamoe and scripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mamoe.configuration_mamoe import MAMoEConfig
from mamoe.model import MAMoEForConditionalGeneration as JaxModel
from scripts.pytorch.pytorch_model import MAMoEForConditionalGeneration as TorchModel
from export_pytorch import convert_jax_to_pytorch

def test_numerical_parity():
    print("Testing JAX vs PyTorch Numerical Parity...")
    config = MAMoEConfig(
        vocab_size=1024,
        hidden_size=64,
        embed_dim=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=32,
        intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        memory_capacity=100,
        memory_dim=64
    )
    
    # 1. Initialize JAX Model
    print("Initializing JAX model...")
    rng = jax.random.PRNGKey(0)
    jax_model = JaxModel(config=config)
    
    batch_size = 2
    enc_seq_len = 16
    dec_seq_len = 16
    
    input_ids = jnp.ones((batch_size, enc_seq_len), dtype=jnp.int32)
    decoder_input_ids = jnp.ones((batch_size, dec_seq_len), dtype=jnp.int32)
    is_eos = jnp.ones((batch_size,), dtype=jnp.int32)
    
    # Init variables
    variables = jax_model.init(
        rng, 
        input_ids=input_ids, 
        decoder_input_ids=decoder_input_ids,
        attention_mask=jnp.ones((batch_size, enc_seq_len), dtype=jnp.int32),
        decoder_attention_mask=jnp.ones((batch_size, dec_seq_len), dtype=jnp.int32),
        is_eos=is_eos
    )
    
    params = variables['params']
    
    # Run JAX forward
    print("Running JAX forward pass...")
    jax_logits, read_prob, write_prob, aux_loss, avg_f_i = jax_model.apply(
        variables, 
        input_ids=input_ids, 
        decoder_input_ids=decoder_input_ids,
        attention_mask=jnp.ones((batch_size, enc_seq_len), dtype=jnp.int32),
        decoder_attention_mask=jnp.ones((batch_size, dec_seq_len), dtype=jnp.int32),
        is_eos=is_eos,
        mutable=False
    )
    
    # 2. Convert to PyTorch
    print("Converting weights to PyTorch...")
    torch_model = TorchModel(config)
    state_dict = convert_jax_to_pytorch(params, torch_model, config)
    torch_model.eval()
    
    # Run PyTorch forward
    print("Running PyTorch forward pass...")
    t_input_ids = torch.ones((batch_size, enc_seq_len), dtype=torch.long)
    t_decoder_input_ids = torch.ones((batch_size, dec_seq_len), dtype=torch.long)
    t_attention_mask = torch.ones((batch_size, enc_seq_len), dtype=torch.long)
    t_decoder_attention_mask = torch.ones((batch_size, dec_seq_len), dtype=torch.long)
    t_is_eos = True
    t_mem_state = torch_model.memory_bank.init_state(bsz=batch_size)
    
    with torch.no_grad():
        t_logits, t_mem_state, t_write_prob = torch_model(
            input_ids=t_input_ids,
            decoder_input_ids=t_decoder_input_ids,
            attention_mask=t_attention_mask,
            decoder_attention_mask=t_decoder_attention_mask,
            mem_state=t_mem_state,
            is_eos=t_is_eos
        )
        
    # 3. Compare Results
    jax_logits_np = np.array(jax_logits)
    torch_logits_np = t_logits.numpy()
    
    max_diff = np.max(np.abs(jax_logits_np - torch_logits_np))
    print(f"Max absolute difference in logits: {max_diff}")
    
    if max_diff < 1e-4:
        print("✅ Parity Test PASSED!")
    else:
        print("❌ Parity Test FAILED!")
        
    assert max_diff < 1e-4, f"Max difference {max_diff} exceeds threshold 1e-4"

if __name__ == "__main__":
    test_numerical_parity()
