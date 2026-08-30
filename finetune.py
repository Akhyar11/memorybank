import os
import time
import jax
import jax.numpy as jnp
import optax
import numpy as np

# Optimasi Tensor Core untuk GPU (T4/P100)
jax.config.update('jax_default_matmul_precision', 'bfloat16')

from typing import Any
from flax.training import train_state
from flax.jax_utils import replicate, unreplicate
import functools

from mamoe.config import MAMoEConfig
from mamoe.model import MAMoEForCausalLM

class MAMoETrainState(train_state.TrainState):
    pass

def create_train_state(rng, config, model, dummy_input):
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    tx = optax.adamw(learning_rate=5e-5, weight_decay=0.1)
    
    state = MAMoETrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    memory_state = variables.get('memory', {})
    
    return state, memory_state

def get_empty_memory_state(rng, config, model, dummy_input):
    variables = model.init(rng, dummy_input)
    return variables.get('memory', {})

@functools.partial(jax.pmap, axis_name='batch')
def finetune_step(state, memory_state, batch_inputs):
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
        total_loss = mean_ce_loss + (0.01 * aux_loss)
        
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
    import pandas as pd
    from tokenizers import Tokenizer
    
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}.")
        return
            
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print(f"Loading chat dataset from {file_path}")
    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}. Generating dummy chat data for test.")
        yield np.random.randint(0, 32000, size=(batch_size, seq_len), dtype=np.uint16)
        return
            
    df = pd.read_parquet(file_path)
    
    text_column = "text"
    for col in ["messages", "text", "conversation", "prompt"]:
        if col in df.columns:
            text_column = col
            break
            
    texts = df[text_column].dropna().astype(str).tolist()
    
    # 1 full epoch only
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
    
    print("-> (Placeholder) Loaded Phase 1 Pre-trained Weights")

    # Pre-compute empty memory template ONCE
    empty_memory_template = jax.tree_util.tree_map(jnp.zeros_like, memory_state)
    
    state        = replicate(state)
    memory_state = replicate(memory_state)
    empty_memory_replicated = replicate(empty_memory_template)
    
    dataset_path = '/kaggle/input/datasets/akhyarsafrudin/dataset-chat/t5gemma2_chat_multiturn.parquet'
    tokenizer_path = '/kaggle/input/datasets/akhyarsafrudin/tokenizer/tokenizer.json'
    
    if not os.path.exists(dataset_path) and os.path.exists('data/raw/t5gemma2_chat_multiturn.parquet'):
        dataset_path = 'data/raw/t5gemma2_chat_multiturn.parquet'
        tokenizer_path = 'tokenizer/tokenizer.json'
        
    dataloader = chat_data_generator(dataset_path, tokenizer_path, total_batch_size, seq_len)
    
    reset_interval = 2 
    
    print(f"\nStarting Distributed Fine-Tuning Loop (1 Epoch)...")
    
    start_time = time.time()
    last_log_time = start_time
    total_tokens = 0
    
    for step, batch in enumerate(dataloader, 1):
        sharded_batch = batch.reshape((num_devices, local_batch_size, seq_len))
        
        state, memory_state, metrics = finetune_step(state, memory_state, sharded_batch)
        
        total_tokens += total_batch_size * seq_len
        
        if step % 10 == 0:
            current_time = time.time()
            elapsed_since_log = current_time - last_log_time
            tokens_per_sec = (total_batch_size * seq_len * 10) / elapsed_since_log
            
            loss_val = unreplicate(metrics['loss'])
            ce_val = unreplicate(metrics['ce_loss'])
            aux_val = unreplicate(metrics['aux_loss'])
            
            elapsed_m = int(current_time - start_time) // 60
            elapsed_s = int(current_time - start_time) % 60
            
            print(f"Step {step:05d} | CE Loss: {ce_val:.4f} | Aux Loss: {aux_val:.4f} | Speed: {tokens_per_sec:,.0f} tok/s | Elapsed: {elapsed_m}m {elapsed_s}s")
            last_log_time = current_time
        
        if step % reset_interval == 0:
            memory_state = empty_memory_replicated
            
    print("\nPhase 2 Fine-Tuning Execution Complete! 🚀")

if __name__ == "__main__":
    main()
