import jax
import jax.numpy as jnp
import flax.linen as nn

def apply_rotary_emb(x, freqs_cis):
    # x shape: (batch_size, seq_len, num_heads, head_dim)
    # freqs_cis shape: (seq_len, head_dim // 2)
    
    # We reshape x to (..., head_dim // 2, 2) to apply complex multiplication
    x_shape = x.shape
    x_reshaped = x.reshape(*x_shape[:-1], -1, 2)
    x_complex = x_reshaped[..., 0] + 1j * x_reshaped[..., 1]
    
    # Ensure freqs_cis is broadcastable
    # freqs_cis shape: (batch_size, seq_len, head_dim // 2)
    # Add head dimension
    freqs_cis = jnp.expand_dims(freqs_cis, axis=2) # (batch_size, seq_len, 1, head_dim // 2)
    
    x_out_complex = x_complex * freqs_cis
    
    x_out = jnp.stack([jnp.real(x_out_complex), jnp.imag(x_out_complex)], axis=-1)
    return x_out.reshape(*x_shape)

class RoPE(nn.Module):
    dim: int
    max_position_embeddings: int = 2048
    base: float = 10000.0

    def setup(self):
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        self.freqs_cis = jnp.exp(1j * freqs) # (max_position_embeddings, dim // 2)

    def __call__(self, positions):
        # positions: (batch_size, seq_len)
        return self.freqs_cis[positions]
