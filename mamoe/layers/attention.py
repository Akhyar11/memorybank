import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional
from .embeddings import apply_rotary_emb

class CausalSelfAttention(nn.Module):
    config: any

    @nn.compact
    def __call__(self, hidden_states, freqs_cis, attention_mask: Optional[jax.Array] = None):
        batch_size, seq_len, _ = hidden_states.shape
        num_heads = self.config.num_attention_heads
        head_dim = self.config.head_dim
        
        q_proj = nn.Dense(num_heads * head_dim, use_bias=False, name='q_proj')
        k_proj = nn.Dense(num_heads * head_dim, use_bias=False, name='k_proj')
        v_proj = nn.Dense(num_heads * head_dim, use_bias=False, name='v_proj')
        o_proj = nn.Dense(self.config.hidden_size, use_bias=False, name='o_proj')

        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)

        query_states = query_states.reshape(batch_size, seq_len, num_heads, head_dim)
        key_states = key_states.reshape(batch_size, seq_len, num_heads, head_dim)
        value_states = value_states.reshape(batch_size, seq_len, num_heads, head_dim)

        # Apply RoPE
        query_states = apply_rotary_emb(query_states, freqs_cis)
        key_states = apply_rotary_emb(key_states, freqs_cis)

        # Transpose for attention: (batch_size, num_heads, seq_len, head_dim)
        query_states = jnp.transpose(query_states, (0, 2, 1, 3))
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))

        # Scaled dot-product attention
        attn_weights = jnp.matmul(query_states, jnp.transpose(key_states, (0, 1, 3, 2))) / jnp.sqrt(head_dim)

        # Apply causal mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        causal_mask = jnp.expand_dims(jnp.expand_dims(causal_mask, axis=0), axis=0) # (1, 1, seq_len, seq_len)
        
        attn_weights = jnp.where(causal_mask == 0, -1e9, attn_weights)

        if attention_mask is not None:
            # Combine with attention mask (e.g. for padding)
            # attention_mask shape: (batch_size, 1, 1, seq_len)
            attn_weights = attn_weights + attention_mask

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Optional dropout could be added here
        
        attn_output = jnp.matmul(attn_weights, value_states)

        # Restore original shape: (batch_size, seq_len, num_heads * head_dim)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, num_heads * head_dim)

        # Output projection
        attn_output = o_proj(attn_output)

        return attn_output
