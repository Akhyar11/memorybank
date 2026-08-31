import jax
import jax.numpy as jnp
from mamoe.model import MAMoEForCausalLM
from mamoe.config import MAMoEConfig
from train import create_train_state

config = MAMoEConfig(freeze_embeddings=True)
model = MAMoEForCausalLM(config)
rng = jax.random.PRNGKey(0)

print("Testing with batch size 4...")
dummy_input = jnp.ones((4, 10), dtype=jnp.int32)

state, mem = create_train_state(rng, model, dummy_input)
print("Success!")
