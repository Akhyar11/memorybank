import os
import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Any
from flax.training import train_state
from flax.jax_utils import replicate, unreplicate
import functools

from mamoe.config import MAMoEConfig
from mamoe.model import MAMoEForCausalLM

class MAMoETrainState(train_state.TrainState):
    pass

def create_train_state(rng, config, model, dummy_input):
    """Initializes model parameters and optimizer state."""
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    tx = optax.adamw(learning_rate=3e-4, weight_decay=0.1)
    
    state = MAMoETrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    memory_state = variables.get('memory', {})
    
    return state, memory_state

def get_empty_memory_state(rng, config, model, dummy_input):
    """Returns a fresh, zeroed-out memory state dictionary."""
    variables = model.init(rng, dummy_input)
    return variables.get('memory', {})

@functools.partial(jax.pmap, axis_name='batch')
def train_step(state, memory_state, batch_inputs):
    """Executes a parallel training step across multiple devices."""
    
    batch_size, seq_len = batch_inputs.shape
    labels = jnp.roll(batch_inputs, shift=-1, axis=1)
    labels = labels.at[:, -1].set(0)
    
    def loss_fn(params):
        (logits, read_prob, write_prob, aux_loss), mutated_vars = state.apply_fn(
            {'params': params, 'memory': memory_state},
            batch_inputs,
            mutable=['memory']
        )
        
        vocab_size = logits.shape[-1]
        labels_one_hot = jax.nn.one_hot(labels, vocab_size)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce_loss = -jnp.sum(labels_one_hot * log_probs, axis=-1)
        
        mean_ce_loss = jnp.mean(ce_loss)
        total_loss = mean_ce_loss + aux_loss
        new_memory_state = mutated_vars.get('memory', {})
        
        return total_loss, (mean_ce_loss, aux_loss, new_memory_state)
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, aux_data), grads = grad_fn(state.params)
    
    # Cross-device gradient synchronization (All-Reduce)
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    mean_ce_loss, aux_loss, new_memory_state = aux_data
    
    # Synchronize metrics
    total_loss = jax.lax.pmean(total_loss, axis_name='batch')
    mean_ce_loss = jax.lax.pmean(mean_ce_loss, axis_name='batch')
    aux_loss = jax.lax.pmean(aux_loss, axis_name='batch')
    
    state = state.apply_gradients(grads=grads)
    
    metrics = {
        'loss': total_loss,
        'ce_loss': mean_ce_loss,
        'aux_loss': aux_loss,
    }
    
    return state, new_memory_state, metrics

def data_generator(file_path, tokenizer_path, batch_size, seq_len):
    import pandas as pd
    from tokenizers import Tokenizer
    
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}.")
        while True:
            yield np.random.randint(0, 32000, size=(batch_size, seq_len), dtype=np.uint16)
            
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}. Generating dummy data for test.")
        while True:
            yield np.random.randint(0, 32000, size=(batch_size, seq_len), dtype=np.uint16)
            
    print(f"Streaming dataset from {file_path}...")
    while True:
        # Stream CSV to avoid RAM OOM
        for chunk in pd.read_csv(file_path, chunksize=10000):
            text_column = "text"
            for col in ["text", "prompt", "content", "completion", "text_clean", "article"]:
                if col in chunk.columns:
                    text_column = col
                    break
                    
            texts = chunk[text_column].dropna().astype(str).tolist()
            encoded = tokenizer.encode_batch(texts)
            
            all_tokens = []
            for enc in encoded:
                all_tokens.extend(enc.ids)
                
            total_chunks = len(all_tokens) // seq_len
            all_tokens = np.array(all_tokens[:total_chunks * seq_len], dtype=np.uint16)
            all_tokens = all_tokens.reshape((total_chunks, seq_len))
            
            # Shuffle chunks internally
            np.random.shuffle(all_tokens)
            
            for i in range(0, total_chunks, batch_size):
                if i + batch_size <= total_chunks:
                    yield all_tokens[i:i+batch_size]

def main():
    num_devices = jax.device_count()
    print(f"Number of available devices (GPUs/TPUs/CPUs): {num_devices}")
    
    config = MAMoEConfig()
    model = MAMoEForCausalLM(config=config)
    rng = jax.random.PRNGKey(42)
    
    # Total batch size must be divisible by num_devices
    total_batch_size = max(2, num_devices * 2) 
    local_batch_size = total_batch_size // num_devices
    seq_len = 1024
    
    dummy_input = jnp.ones((local_batch_size, seq_len), dtype=jnp.int32)
    
    print("Initializing Model and Optimizer...")
    state, memory_state = create_train_state(rng, config, model, dummy_input)
    
    # Replicate state across all devices
    state = replicate(state)
    memory_state = replicate(memory_state)
    
    dataset_path = '/kaggle/input/vqfat-indonesian-corpus/vqfat_cosmopedia_id.csv'
    tokenizer_path = 'tokenizer/tokenizer.json'
    
    # We still use local paths as fallback for testing
    if not os.path.exists(dataset_path) and os.path.exists('data/raw/vqfat_cosmopedia_id.csv'):
        dataset_path = 'data/raw/vqfat_cosmopedia_id.csv'
        
    dataloader = data_generator(dataset_path, tokenizer_path, total_batch_size, seq_len)
    
    num_steps = 20
    reset_interval = 4
    
    print(f"\nStarting Distributed Training Loop for {num_steps} steps...")
    print(f"Total Batch Size: {total_batch_size} | Local Batch Size: {local_batch_size}")
    
    for step in range(1, num_steps + 1):
        # Fetch Total Batch: (total_batch_size, seq_len)
        batch = next(dataloader)
        
        # Reshape to (num_devices, local_batch_size, seq_len)
        sharded_batch = batch.reshape((num_devices, local_batch_size, seq_len))
        
        # Train Step across all devices
        state, memory_state, metrics = train_step(state, memory_state, sharded_batch)
        
        # Metrics are already synchronized (pmean), so we can just grab the 0th device's value
        loss_val = unreplicate(metrics['loss'])
        ce_val = unreplicate(metrics['ce_loss'])
        aux_val = unreplicate(metrics['aux_loss'])
        
        print(f"Step {step:03d} | Total Loss: {loss_val:.4f} | CE Loss: {ce_val:.4f} | Aux Loss: {aux_val:.4f}")
        
        # Reset memory
        if step % reset_interval == 0:
            print(f"  [MEMORY] End of conversation. Resetting Memory Bank on all devices...")
            empty_memory = get_empty_memory_state(rng, config, model, dummy_input)
            memory_state = replicate(empty_memory)
            
    print("\nTraining Loop Execution Complete! 🚀")

if __name__ == "__main__":
    main()
