import jax
import jax.numpy as jnp
import flax.linen as nn

class MemoryController(nn.Module):
    """
    Independent gates for READ and WRITE operating on h_EOS.
    Outputs independent probabilities: read_prob, write_prob.
    """
    config: any
    
    @nn.compact
    def __call__(self, h_eos):
        # h_eos shape: (batch_size, hidden_size)
        read_proj = nn.Dense(1, name='read_gate')
        write_proj = nn.Dense(1, name='write_gate')
        
        read_logits = read_proj(h_eos)
        write_logits = write_proj(h_eos)
        
        # Squeeze the last dimension so shape is (batch_size,)
        read_prob = jax.nn.sigmoid(jnp.squeeze(read_logits, axis=-1))
        write_prob = jax.nn.sigmoid(jnp.squeeze(write_logits, axis=-1))
        
        return read_prob, write_prob
