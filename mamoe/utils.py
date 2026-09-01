import jax
import jax.numpy as jnp

def extract_last_valid_hidden(hidden_states: jax.Array, attention_mask: jax.Array = None) -> jax.Array:
    """
    Extracts the last valid hidden state based on the attention mask.
    If attention_mask is None, it extracts the last token in the sequence.
    
    Args:
        hidden_states: (batch_size, seq_len, hidden_size)
        attention_mask: (batch_size, seq_len) with 1s for valid tokens and 0s for padding
    """
    if attention_mask is None:
        return hidden_states[:, -1, :]
    
    # Find the index of the last 1 in the attention mask
    # sequence_lengths will be (batch_size,)
    sequence_lengths = jnp.sum(attention_mask, axis=-1) - 1
    # Ensure it's at least 0 to avoid out-of-bounds error if sequence is completely empty
    sequence_lengths = jnp.maximum(sequence_lengths, 0)
    
    # Gather the hidden state for each sequence
    batch_size = hidden_states.shape[0]
    batch_indices = jnp.arange(batch_size)
    
    last_valid_hidden = hidden_states[batch_indices, sequence_lengths.astype(jnp.int32), :]
    return last_valid_hidden

def make_key_padding_mask(attention_mask: jax.Array, dtype=jnp.float32) -> jax.Array:
    """
    Converts a boolean/int mask (B, S) into an additive attention mask (B, 1, 1, S)
    where valid tokens are 0.0 and padded tokens are -1e9.
    """
    if attention_mask is None:
        return None
    # attention_mask is 1 for valid, 0 for pad
    mask = 1.0 - attention_mask
    mask = mask * -1e9
    # Expand to (B, 1, 1, S) to broadcast over num_heads and seq_len
    return jnp.expand_dims(jnp.expand_dims(mask, axis=1), axis=1).astype(dtype)

def make_causal_mask(seq_len: int, dtype=jnp.float32) -> jax.Array:
    """
    Creates a causal additive mask (1, 1, S, S)
    """
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    mask = jnp.where(mask == 1, 0.0, -1e9)
    return jnp.expand_dims(jnp.expand_dims(mask, axis=0), axis=0).astype(dtype)

