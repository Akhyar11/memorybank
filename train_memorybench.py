import os
import sys
import time
import json
import random

import jax
import jax.numpy as jnp
import optax
import numpy as np
import functools
import orbax.checkpoint as ocp

jax.config.update('jax_default_matmul_precision', 'bfloat16')

from flax.training import train_state
from flax.jax_utils import replicate, unreplicate
from mamoe.config import MAMoEConfig
from mamoe.model import MAMoEForCausalLM

KAGGLE_JSONL    = '/kaggle/input/datasets/akhyarsafrudin/dataset-chat/memorybench_train_samples.jsonl'
LOCAL_JSONL     = 'data/raw/memorybench_train_samples.jsonl'
COLAB_JSONL     = '/content/drive/MyDrive/Colab Notebooks/dataset/memorybench_train_samples.jsonl'

KAGGLE_TOK      = 'tokenizer_hf/tokenizer.json'
LOCAL_TOK       = 'tokenizer_hf/tokenizer.json'

SEQ_LEN          = 1024
LOCAL_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LOG_INTERVAL     = 10
NUM_EPOCHS       = 3

class MAMoETrainState(train_state.TrainState):
    pass

def create_train_state(rng, model, dummy_input):
    import flax
    import flax.traverse_util
    
    dummy_eos = jnp.zeros((dummy_input.shape[0],), dtype=jnp.int32)
    variables    = model.init(rng, dummy_input, attention_mask=None, is_eos=dummy_eos)
    params       = variables['params']
    memory_state = variables.get('memory', {})
    
    if os.path.exists("pretrained_embeds.npy") and getattr(model.config, 'freeze_embeddings', False):
        embeds = np.load("pretrained_embeds.npy")
        params = flax.core.unfreeze(params)
        params['embed_tokens']['embedding'] = jnp.array(embeds)
        params = flax.core.freeze(params)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=5e-5, warmup_steps=50, decay_steps=5_000, end_value=5e-6,
    )
    
    if getattr(model.config, 'freeze_embeddings', False):
        partition_optimizers = {'frozen': optax.set_to_zero(), 'trainable': optax.adamw(lr_schedule, weight_decay=0.1)}
        def map_params(path, _):
            if 'embed_tokens' in path: return 'frozen'
            return 'trainable'
        flat_params = flax.traverse_util.flatten_dict(params)
        param_labels = flax.traverse_util.unflatten_dict({k: map_params(k, v) for k, v in flat_params.items()})
        param_labels = flax.core.freeze(param_labels)
        base_tx = optax.multi_transform(partition_optimizers, param_labels)
    else:
        base_tx = optax.adamw(lr_schedule, weight_decay=0.1)
        
    tx = optax.chain(optax.clip_by_global_norm(1.0), base_tx)
    tx = optax.MultiSteps(tx, every_k_schedule=GRAD_ACCUM_STEPS)
    
    return MAMoETrainState.create(apply_fn=model.apply, params=params, tx=tx), memory_state

@functools.partial(jax.pmap, axis_name='batch')
def finetune_step(state, memory_state, batch_inputs):
    labels = jnp.roll(batch_inputs, shift=-1, axis=1).at[:, -1].set(0)

    def loss_fn(params):
        (logits, _, _, aux_loss, avg_f_i), mutated = state.apply_fn(
            {'params': params, 'memory': memory_state}, batch_inputs, mutable=['memory'],
        )
        vocab_size = logits.shape[-1]
        log_probs  = jax.nn.log_softmax(logits, axis=-1)
        ce_loss    = -jnp.sum(jax.nn.one_hot(labels, vocab_size) * log_probs, axis=-1)
        loss_mask  = (labels != 0).astype(jnp.float32)
        mean_ce    = jnp.sum(ce_loss * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)
        return mean_ce + 0.01 * aux_loss, (mean_ce, aux_loss, avg_f_i, mutated.get('memory', {}))

    (total_loss, (ce_loss, aux_loss, avg_f_i, new_mem)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads      = jax.lax.pmean(grads,      axis_name='batch')
    ce_loss    = jax.lax.pmean(ce_loss,    axis_name='batch')
    aux_loss   = jax.lax.pmean(aux_loss,   axis_name='batch')
    avg_f_i    = jax.lax.pmean(avg_f_i,    axis_name='batch')

    state = state.apply_gradients(grads=grads)
    return state, new_mem, {'ce_loss': ce_loss, 'aux_loss': aux_loss, 'expert_load': avg_f_i}

@functools.partial(jax.pmap, axis_name='batch')
def eval_step(state, memory_state, batch_inputs):
    labels = jnp.roll(batch_inputs, shift=-1, axis=1).at[:, -1].set(0)
    (logits, _, _, aux_loss, _), mutated = state.apply_fn(
        {'params': state.params, 'memory': memory_state}, batch_inputs, mutable=['memory'],
    )
    vocab_size = logits.shape[-1]
    log_probs  = jax.nn.log_softmax(logits, axis=-1)
    ce_loss    = -jnp.sum(jax.nn.one_hot(labels, vocab_size) * log_probs, axis=-1)
    loss_mask  = (labels != 0).astype(jnp.float32)
    mean_ce    = jnp.sum(ce_loss * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)
    
    ce_loss  = jax.lax.pmean(mean_ce, axis_name='batch')
    aux_loss = jax.lax.pmean(aux_loss, axis_name='batch')
    return mutated.get('memory', {}), {'ce_loss': ce_loss, 'aux_loss': aux_loss}

def load_data(path):
    texts = []
    print(f"Loading {path} into memory...")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            text_col = next((c for c in ["messages", "text", "conversation", "prompt", "output", "content"] if c in row), list(row.keys())[0])
            val = row.get(text_col)
            if val is None and row: val = list(row.values())[0]
            if isinstance(val, list):
                try: val = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', msg.get('text', ''))}" for msg in val if isinstance(msg, dict)])
                except: val = str(val)
            texts.append(str(val))
    
    random.seed(42)
    random.shuffle(texts)
    split_idx = int(0.9 * len(texts))
    return texts[:split_idx], texts[split_idx:]

def conversation_generator(texts, tok_path, total_batch_size, seq_len):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tok_path)
    buffer = []
    
    for content in texts:
        tokens = tokenizer.encode(content).ids
        buffer.append(tokens)
        if len(buffer) == total_batch_size:
            max_len = max(len(ids) for ids in buffer)
            padded_len = ((max_len + seq_len - 1) // seq_len) * seq_len
            padded = np.zeros((total_batch_size, padded_len), dtype=np.int32)
            for i, ids in enumerate(buffer): padded[i, :len(ids)] = ids
            for chunk_start in range(0, padded_len, seq_len):
                yield padded[:, chunk_start : chunk_start + seq_len], False
            yield None, True
            buffer = []

def resolve_paths():
    tok = LOCAL_TOK
    path = None
    if os.path.exists(KAGGLE_JSONL): path, tok = KAGGLE_JSONL, KAGGLE_TOK
    elif os.path.exists(LOCAL_JSONL): path = LOCAL_JSONL
    elif os.path.exists(COLAB_JSONL): path = COLAB_JSONL
    return path, tok

def main():
    if 'COLAB_TPU_ADDR' in os.environ:
        import jax.tools.colab_tpu as colab_tpu
        colab_tpu.setup_tpu()

    num_devices = jax.device_count()
    total_batch_size = LOCAL_BATCH_SIZE * num_devices
    
    data_path, tok_path = resolve_paths()
    if not data_path:
        print("❌ Dataset tidak ditemukan!")
        sys.exit(1)
        
    train_texts, val_texts = load_data(data_path)
    print(f"Split: {len(train_texts)} train, {len(val_texts)} validation.")

    config = MAMoEConfig()
    model  = MAMoEForCausalLM(config=config)
    rng    = jax.random.PRNGKey(0)
    
    dummy = jnp.ones((LOCAL_BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
    state, memory_state = create_train_state(rng, model, dummy)
    
    ckpt_dir = '/kaggle/working/checkpoints/phase1' if os.path.exists('/kaggle') else 'checkpoints/phase1'
    if os.path.exists(ckpt_dir):
        print(f"Loading Phase 1 weights from {ckpt_dir}...")
        state = ocp.StandardCheckpointer().restore(os.path.abspath(ckpt_dir), target=state)
        print("✅ Pre-trained Weights Loaded!")
        
    state = replicate(state)
    memory_state = replicate(memory_state)
    val_memory_state = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), memory_state)

    print(f"Starting Training on MemoryBench ({NUM_EPOCHS} Epochs)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== EPOCH {epoch}/{NUM_EPOCHS} ==========")
        
        # --- TRAIN ---
        train_loader = conversation_generator(train_texts, tok_path, total_batch_size, SEQ_LEN)
        step = 0
        last_log = time.time()
        for batch, should_reset in train_loader:
            if should_reset:
                memory_state = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), memory_state)
                continue
            
            step += 1
            sharded = batch.reshape((num_devices, LOCAL_BATCH_SIZE, SEQ_LEN))
            state, memory_state, metrics = finetune_step(state, memory_state, sharded)
            
            if step % LOG_INTERVAL == 0:
                ce = float(unreplicate(metrics['ce_loss']))
                print(f"[Train] Ep {epoch} Step {step} | CE {ce:.4f} | Speed: {(total_batch_size*SEQ_LEN*LOG_INTERVAL)/(time.time()-last_log):.0f} tok/s")
                last_log = time.time()
                
        # --- EVAL ---
        print(f"\nRunning Validation...")
        val_loader = conversation_generator(val_texts, tok_path, total_batch_size, SEQ_LEN)
        val_ce, val_steps = 0.0, 0
        for batch, should_reset in val_loader:
            if should_reset:
                val_memory_state = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), val_memory_state)
                continue
            sharded = batch.reshape((num_devices, LOCAL_BATCH_SIZE, SEQ_LEN))
            val_memory_state, metrics = eval_step(state, val_memory_state, sharded)
            val_ce += float(unreplicate(metrics['ce_loss']))
            val_steps += 1
            
        final_val_ce = val_ce / max(val_steps, 1)
        print(f"✅ Epoch {epoch} Validation CE Loss: {final_val_ce:.4f}\n")

    out_dir = '/kaggle/working/checkpoints/memorybench' if os.path.exists('/kaggle') else 'checkpoints/memorybench'
    os.makedirs(out_dir, exist_ok=True)
    ocp.StandardCheckpointer().save(os.path.abspath(out_dir), unreplicate(state), force=True)
    print(f"✅ Saved to {out_dir}")

if __name__ == '__main__':
    main()
