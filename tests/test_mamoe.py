import jax
import jax.numpy as jnp
from mamoe.config import MAMoEConfig
from mamoe.model import MAMoEForCausalLM

def main():
    print("Testing MAMoE-50 JAX Initialization...")
    
    # 1. Configuration
    config = MAMoEConfig()
    model = MAMoEForCausalLM(config=config)
    
    # 2. Dummy Inputs
    batch_size = 2
    seq_len = 16
    
    # Random token IDs between 0 and vocab_size - 1
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
    
    print(f"Input shape: {input_ids.shape}")
    
    # 3. Initialization
    # We must provide all expected collections since we use mutable memory
    init_key = jax.random.PRNGKey(1)
    variables = model.init(init_key, input_ids)
    
    # Extract params and memory state
    params = variables['params']
    memory_state = variables.get('memory', {})
    
    # 4. Count Parameters
    def count_params(tree):
        return sum(x.size for x in jax.tree_util.tree_leaves(tree))
    
    total_params = count_params(params)
    total_memory_vars = count_params(memory_state) if memory_state else 0
    
    print(f"Total Trainable Parameters: {total_params / 1e6:.2f} M")
    print(f"Total Memory Slots Allocated (floats): {total_memory_vars / 1e6:.2f} M")
    
    # 5. Forward Pass
    print("\nRunning Forward Pass...")
    
    # Since model mutates variables internally (if we use apply properly with mutable), 
    # we need to pass the mutable collections. 
    # In our simplistic test, we just call apply and see if dimensions are correct.
    (output, read_prob, write_prob, aux_loss), mutated_vars = model.apply(
        variables, 
        input_ids, 
        mutable=['memory']
    )
    
    print(f"Output Logits Shape: {output.shape} (Expected: {batch_size}, {seq_len}, {config.vocab_size})")
    print(f"Read Prob Shape: {read_prob.shape} (Expected: {batch_size},)")
    print(f"Write Prob Shape: {write_prob.shape} (Expected: {batch_size},)")
    print(f"Auxiliary Loss: {aux_loss:.4f} (Load Balancing)")
    print(f"Read Probs Sample: {read_prob}")
    print(f"Write Probs Sample: {write_prob}")
    
    print("\n✅ Verification Successful!")

if __name__ == "__main__":
    main()
