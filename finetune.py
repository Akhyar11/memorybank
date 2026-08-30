import os
import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
import functools
import orbax.checkpoint as ocp

# ── Optimasi Tensor Core (T4/P100 bfloat16) ─────────────────────────────────
jax.config.update('jax_default_matmul_precision', 'bfloat16')

from flax.training import train_state
from flax.jax_utils import replicate, unreplicate

from mamoe.config import MAMoEConfig
from mamoe.model import MAMoEForCausalLM

# ─── Path Konfigurasi ────────────────────────────────────────────────────────
KAGGLE_PARQUET  = '/kaggle/input/datasets/akhyarsafrudin/dataset-chat/t5gemma2_chat_multiturn.parquet'
KAGGLE_TOK      = '/kaggle/input/datasets/akhyarsafrudin/tokenizer/tokenizer.json'
LOCAL_PARQUET   = 'data/raw/t5gemma2_chat_multiturn.parquet'
LOCAL_TOK       = 'tokenizer/tokenizer.json'
# ─────────────────────────────────────────────────────────────────────────────

# ─── Hyperparameters ─────────────────────────────────────────────────────────
SEQ_LEN          = 1024
LOCAL_BATCH_SIZE = 4      # per device
GRAD_ACCUM_STEPS = 4      # Akumulasi 4 step (Total Effective Batch = 32)
LOG_INTERVAL     = 10
# ─────────────────────────────────────────────────────────────────────────────

class MAMoETrainState(train_state.TrainState):
    pass

# ── Model Init ───────────────────────────────────────────────────────────────
def create_train_state(rng, model, dummy_input):
    import os
    import numpy as np
    import flax
    import flax.traverse_util
    
    dummy_eos = jnp.zeros((1,), dtype=jnp.int32)
    variables    = model.init(rng, dummy_input, attention_mask=None, is_eos=dummy_eos)
    params       = variables['params']
    memory_state = variables.get('memory', {})
    
    if os.path.exists("pretrained_embeds.npy") and getattr(model.config, 'freeze_embeddings', False):
        print("Injecting pretrained embeddings...")
        embeds = np.load("pretrained_embeds.npy")
        params = flax.core.unfreeze(params)
        params['embed_tokens']['embedding'] = jnp.array(embeds)
        params = flax.core.freeze(params)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=5e-5,        # LR lebih kecil untuk fine-tuning
        warmup_steps=100,
        decay_steps=10_000,
        end_value=5e-6,
    )
    
    if getattr(model.config, 'freeze_embeddings', False):
        partition_optimizers = {
            'frozen': optax.set_to_zero(),
            'trainable': optax.adamw(lr_schedule, weight_decay=0.1)
        }
        def map_params(path, _):
            # path is a tuple of strings, e.g., ('embed_tokens', 'embedding')
            if 'embed_tokens' in path: return 'frozen'
            return 'trainable'
        flat_params = flax.traverse_util.flatten_dict(params)
        param_labels = flax.traverse_util.unflatten_dict({k: map_params(k, v) for k, v in flat_params.items()})
        param_labels = flax.core.freeze(param_labels)
        base_tx = optax.multi_transform(partition_optimizers, param_labels)
    else:
        base_tx = optax.adamw(lr_schedule, weight_decay=0.1)
        
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        base_tx,
    )
    # Wrap dengan MultiSteps untuk gradient accumulation
    tx = optax.MultiSteps(tx, every_k_schedule=GRAD_ACCUM_STEPS)
    
    state = MAMoETrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, memory_state

# ── Training Step (pmap) ─────────────────────────────────────────────────────
@functools.partial(jax.pmap, axis_name='batch')
def finetune_step(state, memory_state, batch_inputs):
    labels = jnp.roll(batch_inputs, shift=-1, axis=1).at[:, -1].set(0)

    def loss_fn(params):
        (logits, _, _, aux_loss, avg_f_i), mutated = state.apply_fn(
            {'params': params, 'memory': memory_state},
            batch_inputs,
            mutable=['memory'],
        )
        vocab_size = logits.shape[-1]
        log_probs  = jax.nn.log_softmax(logits, axis=-1)
        ce_loss    = -jnp.sum(jax.nn.one_hot(labels, vocab_size) * log_probs, axis=-1)
        mean_ce    = jnp.mean(ce_loss)
        total_loss = mean_ce + 0.01 * aux_loss
        return total_loss, (mean_ce, aux_loss, avg_f_i, mutated.get('memory', {}))

    (total_loss, (ce_loss, aux_loss, avg_f_i, new_mem)), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(state.params)

    grads      = jax.lax.pmean(grads,      axis_name='batch')
    total_loss = jax.lax.pmean(total_loss, axis_name='batch')
    ce_loss    = jax.lax.pmean(ce_loss,    axis_name='batch')
    aux_loss   = jax.lax.pmean(aux_loss,   axis_name='batch')
    avg_f_i    = jax.lax.pmean(avg_f_i,    axis_name='batch')

    state = state.apply_gradients(grads=grads)
    return state, new_mem, {'loss': total_loss, 'ce_loss': ce_loss, 'aux_loss': aux_loss, 'expert_load': avg_f_i}

# ── Data Loading — Per-Conversation ─────────────────────────────────────────
def conversation_generator(parquet_path, tok_path, total_batch_size, seq_len):
    """
    Yield (batch, should_reset_memory).
    - Memproses total_batch_size conversation secara parallel.
    - Setelah setiap grup conversation selesai → yield (None, True) sebagai sinyal reset memory.
    """
    import pandas as pd
    from tokenizers import Tokenizer
    import pyarrow.parquet as pq

    tokenizer = Tokenizer.from_file(tok_path)
    is_test = os.environ.get('QUICK_TEST') == '1'
    conv_limit = 50 if is_test else float('inf')
    conv_count = 0

    # Auto-detect text column
    df_sample = pd.read_parquet(parquet_path, columns=None, engine='pyarrow')
    text_col = next(
        (c for c in ["messages", "text", "conversation", "prompt", "output", "content"]
         if c in df_sample.columns),
        df_sample.columns[0]
    )
    
    print(f"   Parquet text column: '{text_col}'")

    buffer = []
    parquet_file = pq.ParquetFile(parquet_path)
    
    for batch in parquet_file.iter_batches(batch_size=1000):
        for row in batch.to_pylist():
            if conv_count >= conv_limit:
                if is_test:
                    print("\n⚠️ QUICK_TEST MODE: Reached 50 conversations. Stopping.")
                return
            
            # Simple conversion if list of dicts (for 'messages')
            content = str(row[text_col])
            tokens = tokenizer.encode(content).ids
            buffer.append(tokens)
            
            if len(buffer) == total_batch_size:
                # Pad and yield
                max_len = max(len(ids) for ids in buffer)
                padded_len = ((max_len + seq_len - 1) // seq_len) * seq_len
                padded = np.zeros((total_batch_size, padded_len), dtype=np.int32)
                for i, ids in enumerate(buffer):
                    padded[i, :len(ids)] = ids
                
                for chunk_start in range(0, padded_len, seq_len):
                    chunk = padded[:, chunk_start : chunk_start + seq_len]
                    yield chunk, False
                
                yield None, True
                buffer = []
                conv_count += 1

# ── Path Resolution ─────────────────────────────────────────────────────────
def resolve_paths():
    if os.path.exists(KAGGLE_PARQUET):
        return KAGGLE_PARQUET, KAGGLE_TOK
    if os.path.exists(LOCAL_PARQUET):
        return LOCAL_PARQUET, LOCAL_TOK
    raise FileNotFoundError("Parquet dataset tidak ditemukan!")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    num_devices      = jax.device_count()
    total_batch_size = LOCAL_BATCH_SIZE * num_devices
    print(f"Devices        : {num_devices}")
    print(f"Micro-batch    : {total_batch_size} total ({LOCAL_BATCH_SIZE} per device)")
    print(f"Grad accum     : {GRAD_ACCUM_STEPS} steps")
    print(f"Effective batch: {total_batch_size * GRAD_ACCUM_STEPS} (Safe for 16GB VRAM)")
    print(f"Seq len        : {SEQ_LEN}")
    print()

    config = MAMoEConfig()
    model  = MAMoEForCausalLM(config=config)
    rng    = jax.random.PRNGKey(0)

    dummy  = jnp.ones((LOCAL_BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
    print("Initializing model for Phase 2: Fine-Tuning...")
    state, memory_state = create_train_state(rng, model, dummy)
    
    # --- LOAD PHASE 1 CHECKPOINT ---
    ckpt_dir = '/kaggle/working/checkpoints/phase1' if os.path.exists('/kaggle') else 'checkpoints/phase1'
    if os.path.exists(ckpt_dir):
        print(f"Loading Phase 1 weights from {ckpt_dir}...")
        checkpointer = ocp.StandardCheckpointer()
        # Restore state
        state = checkpointer.restore(os.path.abspath(ckpt_dir), target=state)
        print("✅ Phase 1 Pre-trained Weights Loaded!")
    else:
        print("⚠️ Warning: Phase 1 checkpoint not found, starting from scratch!")

    # Pre-compute empty memory template ONCE
    empty_memory_template = jax.tree_util.tree_map(jnp.zeros_like, memory_state)
    
    state        = replicate(state)
    memory_state = replicate(memory_state)
    print("Done.\n")

    parquet_path, tok_path = resolve_paths()
    dataloader = conversation_generator(parquet_path, tok_path, total_batch_size, SEQ_LEN)

    print("Starting Phase 2: Fine-Tuning (1 Epoch, per-conversation memory reset)...\n")
    start_time    = time.time()
    last_log_time = start_time
    total_tokens  = 0
    step          = 0
    conv_resets   = 0

    for batch, should_reset in dataloader:
        if should_reset:
            # Reset memory di batas antar conversation — inilah yang benar
            memory_state = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), memory_state)
            conv_resets += 1
            continue

        step += 1
        sharded = batch.reshape((num_devices, LOCAL_BATCH_SIZE, SEQ_LEN))
        state, memory_state, metrics = finetune_step(state, memory_state, sharded)
        total_tokens += total_batch_size * SEQ_LEN

        if step % LOG_INTERVAL == 0:
            now           = time.time()
            tok_per_sec   = (total_batch_size * SEQ_LEN * LOG_INTERVAL) / (now - last_log_time)
            elapsed       = int(now - start_time)
            ce_val        = float(unreplicate(metrics['ce_loss']))
            aux_val       = float(unreplicate(metrics['aux_loss']))
            expert_load   = np.array(unreplicate(metrics['expert_load']))
            expert_str    = ' '.join(f'E{i}:{v*100:.0f}%' for i, v in enumerate(expert_load))
            print(
                f"Step {step:06d} | "
                f"CE {ce_val:.4f} | "
                f"Aux {aux_val:.4f} | "
                f"Speed {tok_per_sec:>8,.0f} tok/s | "
                f"Conv resets: {conv_resets} | "
                f"Elapsed {elapsed//60}m {elapsed%60:02d}s"
            )
            print(f"          Expert load: [{expert_str}]")
            last_log_time = now

    total_elapsed = int(time.time() - start_time)
    print(f"\n✅ Phase 2 Complete!  "
          f"Total: {total_elapsed//3600}h {(total_elapsed%3600)//60}m  |  "
          f"Conversations: {conv_resets}  |  "
          f"Tokens: {total_tokens:,}")

    # --- SAVE CHECKPOINT ---
    print("\n💾 Saving Phase 2 (Fine-tuned) checkpoint...")
    ckpt_dir = '/kaggle/working/checkpoints/phase2' if os.path.exists('/kaggle') else 'checkpoints/phase2'
    os.makedirs(ckpt_dir, exist_ok=True)
    
    unreplicated_state = unreplicate(state)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(os.path.abspath(ckpt_dir), unreplicated_state, force=True)
    checkpointer.wait_until_finished()
    print(f"✅ Checkpoint saved to: {ckpt_dir}")

if __name__ == '__main__':
    main()
