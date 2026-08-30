import jax
import jax.numpy as jnp
import flax.linen as nn

def silu(x):
    return jax.nn.silu(x)

class SwiGLU(nn.Module):
    # This module handles the activation. 
    # Usually in Llama-like models, the FFN computes SwiGLU(x) = (x * W1) * SiLU(x * W3) * W2
    # So the activation function itself just takes two inputs or splits a doubled input.
    # We will implement it as an activation that splits the input tensor in half along the last dimension.
    
    @nn.compact
    def __call__(self, x):
        # Assumes x was projected to 2 * intermediate_size
        x1, x2 = jnp.split(x, 2, axis=-1)
        return silu(x1) * x2
