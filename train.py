import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Any
from flax.training import train_state

from mamoe.config import MAMoEConfig
from mamoe.model import MAMoEForCausalLM

# We need a custom TrainState to handle random keys if we use dropout, 
# but for this basic loop, standard TrainState is fine for params.
# Memory state is handled explicitly because it's mutable and not optimized via gradients.
class MAMoETrainState(train_state.TrainState):
    pass

def create_train_state(rng, config, model, dummy_input):
    """Initializes model parameters and optimizer state."""
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    # We use AdamW as standard for modern Transformers
    tx = optax.adamw(learning_rate=3e-4, weight_decay=0.1)
    
    return MAMoETrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    ), variables.get('memory', {})

def get_empty_memory_state(rng, config, model, dummy_input):
    """Returns a fresh, zeroed-out memory state dictionary."""
    variables = model.init(rng, dummy_input)
    return variables.get('memory', {})

@jax.jit
def train_step(state, memory_state, batch_inputs):
    """Executes a single training step (forward, backward, optimize)."""
    
    # In standard causal LM, inputs are shifted to create labels
    # inputs:  [A, B, C, D]
    # labels:  [B, C, D, EOS]
    # For simplicity, we just shift by 1 and pad with 0 for the last token.
    batch_size, seq_len = batch_inputs.shape
    labels = jnp.roll(batch_inputs, shift=-1, axis=1)
    labels = labels.at[:, -1].set(0) # In practice, pad with EOS token id
    
    def loss_fn(params):
        # Forward pass with mutable memory
        (logits, read_prob, write_prob, aux_loss), mutated_vars = state.apply_fn(
            {'params': params, 'memory': memory_state},
            batch_inputs,
            mutable=['memory']
        )
        
        # Standard Cross-Entropy Loss
        # Logits shape: (batch, seq, vocab)
        # Labels shape: (batch, seq)
        vocab_size = logits.shape[-1]
        
        labels_one_hot = jax.nn.one_hot(labels, vocab_size)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce_loss = -jnp.sum(labels_one_hot * log_probs, axis=-1)
        
        # Average over batch and sequence
        mean_ce_loss = jnp.mean(ce_loss)
        
        # Total Loss
        total_loss = mean_ce_loss + aux_loss
        
        new_memory_state = mutated_vars.get('memory', {})
        
        return total_loss, (mean_ce_loss, aux_loss, new_memory_state)
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, aux_data), grads = grad_fn(state.params)
    
    mean_ce_loss, aux_loss, new_memory_state = aux_data
    
    # Update weights
    state = state.apply_gradients(grads=grads)
    
    metrics = {
        'loss': total_loss,
        'ce_loss': mean_ce_loss,
        'aux_loss': aux_loss,
    }
    
    return state, new_memory_state, metrics

def data_generator(file_path, batch_size, seq_len):
    """Yields batches of data from the chunked .npy file."""
    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}. Generating dummy data for test.")
        while True:
            yield np.random.randint(0, 32000, size=(batch_size, seq_len), dtype=np.uint16)
            
    data = np.load(file_path) # Shape should be (N, seq_len)
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    
    while True:
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            if i + batch_size <= num_samples:
                batch_indices = indices[i:i+batch_size]
                yield data[batch_indices]

def main():
    # Setup
    config = MAMoEConfig()
    model = MAMoEForCausalLM(config=config)
    rng = jax.random.PRNGKey(42)
    
    batch_size = 2
    seq_len = 64 # Standard chunk size from prepare_id_dataset.py
    
    # Dummy input for initialization
    dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    print("Initializing Model and Optimizer...")
    state, memory_state = create_train_state(rng, config, model, dummy_input)
    
    # Dataloader
    dataset_path = 'data/pretrain/train_chunks.npy'
    dataloader = data_generator(dataset_path, batch_size, seq_len)
    
    # Training Loop Parameters
    num_steps = 20
    reset_interval = 4  # Reset memory every 4 steps (simulating a 4-turn conversation)
    
    print(f"\nStarting Training Loop for {num_steps} steps...")
    print(f"Memory Bank will be reset every {reset_interval} steps.\n")
    
    for step in range(1, num_steps + 1):
        # 1. Fetch Batch
        batch = next(dataloader)
        
        # 2. Execute Train Step
        state, memory_state, metrics = train_step(state, memory_state, batch)
        
        # 3. Print Metrics
        loss_val = metrics['loss']
        ce_val = metrics['ce_loss']
        aux_val = metrics['aux_loss']
        print(f"Step {step:03d} | Total Loss: {loss_val:.4f} | CE Loss: {ce_val:.4f} | Aux Loss: {aux_val:.4f}")
        
        # 4. Memory Reset Logic (End of Conversation)
        if step % reset_interval == 0:
            print(f"  [MEMORY] End of conversation reached (Step {step}). Resetting Neural Memory Bank to empty state...")
            memory_state = get_empty_memory_state(rng, config, model, dummy_input)
            
    print("\nTraining Loop Execution Complete! 🚀")

if __name__ == "__main__":
    main()
