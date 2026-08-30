import os
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
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
    
    # Lower learning rate for Fine-Tuning
    tx = optax.adamw(learning_rate=5e-5, weight_decay=0.1)
    
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
def finetune_step(state, memory_state, batch_inputs):
    """Executes a parallel fine-tuning step."""
    
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
        
        # In a full setup, you would apply a loss mask here to ignore user prompt tokens
        mean_ce_loss = jnp.mean(ce_loss)
        total_loss = mean_ce_loss + (0.01 * aux_loss) # Lower aux loss weight for FT
        
        new_memory_state = mutated_vars.get('memory', {})
        
        return total_loss, (mean_ce_loss, aux_loss, new_memory_state)
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, aux_data), grads = grad_fn(state.params)
    
    grads = jax.lax.pmean(grads, axis_name='batch')
    mean_ce_loss, aux_loss, new_memory_state = aux_data
    
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

def chat_data_generator(file_path, tokenizer_path, batch_size, seq_len):
    """Streams data from Kaggle Chat Parquet"""
    import pandas as pd
    from tokenizers import Tokenizer
    
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}.")
        while True:
            yield np.random.randint(0, 32000, size=(batch_size, seq_len), dtype=np.uint16)
            
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print(f"Loading chat dataset from {file_path}")
    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}. Generating dummy chat data for test.")
        while True:
            yield np.random.randint(0, 32000, size=(batch_size, seq_len), dtype=np.uint16)
            
    # Load parquet (usually small enough to fit in RAM)
    df = pd.read_parquet(file_path)
    
    # Auto-detect text column
    text_column = "text"
    for col in ["messages", "text", "conversation", "prompt"]:
        if col in df.columns:
            text_column = col
            break
            
    # Convert chat data to text if it's in list format, else just use strings
    texts = df[text_column].dropna().astype(str).tolist()
    
    while True:
        # We process in batches to avoid locking up CPU
        chunk_size = 10000
        for idx in range(0, len(texts), chunk_size):
            chunk_texts = texts[idx : idx+chunk_size]
            encoded = tokenizer.encode_batch(chunk_texts)
            
            all_tokens = []
            for enc in encoded:
                all_tokens.extend(enc.ids)
                
            total_chunks = len(all_tokens) // seq_len
            all_tokens = np.array(all_tokens[:total_chunks * seq_len], dtype=np.uint16)
            all_tokens = all_tokens.reshape((total_chunks, seq_len))
            
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
    
    total_batch_size = max(2, num_devices * 2) 
    local_batch_size = total_batch_size // num_devices
    seq_len = 1024
    
    dummy_input = jnp.ones((local_batch_size, seq_len), dtype=jnp.int32)
    
    print("Initializing Model and Optimizer for Phase 2: Fine-Tuning...")
    state, memory_state = create_train_state(rng, config, model, dummy_input)
    
    # TODO: Load Checkpoint from Phase 1 (Pre-training) here using flax.training.checkpoints
    print("-> (Placeholder) Loaded Phase 1 Pre-trained Weights")
    
    state = replicate(state)
    memory_state = replicate(memory_state)
    
    dataset_path = '/kaggle/input/t5gemma2-indonesia-chat/t5gemma2_chat_multiturn.parquet'
    tokenizer_path = 'tokenizer/tokenizer.json'
    
    # We still use local paths as fallback for testing
    if not os.path.exists(dataset_path) and os.path.exists('data/raw/t5gemma2_chat_multiturn.parquet'):
        dataset_path = 'data/raw/t5gemma2_chat_multiturn.parquet'
        
    dataloader = chat_data_generator(dataset_path, tokenizer_path, total_batch_size, seq_len)
    
    num_steps = 20
    reset_interval = 2 # Reset memory more frequently per short conversation
    
    print(f"\nStarting Distributed Fine-Tuning Loop for {num_steps} steps...")
    
    for step in range(1, num_steps + 1):
        batch = next(dataloader)
        sharded_batch = batch.reshape((num_devices, local_batch_size, seq_len))
        
        state, memory_state, metrics = finetune_step(state, memory_state, sharded_batch)
        
        loss_val = unreplicate(metrics['loss'])
        ce_val = unreplicate(metrics['ce_loss'])
        aux_val = unreplicate(metrics['aux_loss'])
        
        print(f"Step {step:03d} | FT Loss: {loss_val:.4f} | CE Loss: {ce_val:.4f} | Aux Loss: {aux_val:.4f}")
        
        if step % reset_interval == 0:
            print(f"  [MEMORY] End of Chat Turn. Resetting Memory Bank on all devices...")
            empty_memory = get_empty_memory_state(rng, config, model, dummy_input)
            memory_state = replicate(empty_memory)
            
    print("\nPhase 2 Fine-Tuning Execution Complete! 🚀")

if __name__ == "__main__":
    main()
