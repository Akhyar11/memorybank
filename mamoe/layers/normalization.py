import jax
import jax.numpy as jnp
import flax.linen as nn

class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    
    @nn.compact
    def __call__(self, x):
        # x is of shape [..., dim]
        weight = self.param('weight', nn.initializers.ones, (self.dim,))
        
        # Calculate variance
        # Ensure we compute variance in float32 for numerical stability
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True, dtype=jnp.float32)
        
        # Normalize
        x_norm = x * jax.lax.rsqrt(variance + self.eps)
        
        # Scale
        return (x_norm * weight).astype(x.dtype)
