import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys
import time

import threading
import queue
import torch
import jax
import jax.numpy as jnp
import optax
import numpy as np
import functools
import orbax.checkpoint as ocp

from export_pytorch import convert_jax_to_pytorch
from scripts.pytorch.pytorch_model import MAMoEForConditionalGeneration as PyTorchMAMoE, MAMoEConfig as PyTorchConfig

# ── Optimasi Tensor Core (T4/P100 bfloat16) ─────────────────────────────────
jax.config.update('jax_default_matmul_precision', 'bfloat16')

from flax.training import train_state
from flax.jax_utils import replicate, unreplicate

from mamoe.config import MAMoEConfig
from mamoe.model import MAMoEForConditionalGeneration

# ─── Path Konfigurasi ────────────────────────────────────────────────────────
KAGGLE_TOKENS   = '/kaggle/input/datasets/akhyarsafrudin/dataset-chat/vqfat_clean_tokens.npy'
KAGGLE_CSV      = '/kaggle/input/datasets/akhyarsafrudin/dataset-chat/vqfat_clean.csv'
KAGGLE_TOK      = 'tokenizer_hf/tokenizer.json'
LOCAL_TOKENS    = 'data/pretrain/vqfat_clean_tokens.npy'
LOCAL_CSV       = 'data/raw/vqfat_clean.csv'
LOCAL_TOK       = 'tokenizer_hf/tokenizer.json'
COLAB_CSV       = '/content/drive/MyDrive/Colab Notebooks/dataset/vqfat_clean.csv'
# ─────────────────────────────────────────────────────────────────────────────

# ─── Hyperparameters ─────────────────────────────────────────────────────────
SEQ_LEN           = 1024
ENC_SEQ_LEN       = 512
DEC_SEQ_LEN       = 512
LOCAL_BATCH_SIZE  = 4       # per device  → Total = 4 × num_devices
GRAD_ACCUM_STEPS  = 4       # Akumulasi 4 step (Total Effective Batch = 32)
LOG_INTERVAL      = 10
PREFETCH_QUEUE    = 8       # buffer batches di RAM sebelum GPU butuh
NUM_EPOCHS        = 20
# ─────────────────────────────────────────────────────────────────────────────

class MAMoETrainState(train_state.TrainState):
    pass

# ── Model Init ───────────────────────────────────────────────────────────────
def create_train_state(rng, model, dummy_enc_input, dummy_dec_input):
    import os
    import numpy as np
    import flax
    import flax.traverse_util
    
    dummy_eos = jnp.zeros((dummy_enc_input.shape[0],), dtype=jnp.int32)
    variables    = model.init(rng, input_ids=dummy_enc_input, decoder_input_ids=dummy_dec_input, attention_mask=None, is_eos=dummy_eos)
    params         = variables['params']
    memory_state   = variables.get('memory', {})

    if os.path.exists("pretrained_embeds.npy") and getattr(model.config, 'freeze_embeddings', False):
        print("Injecting pretrained embeddings...")
        embeds = np.load("pretrained_embeds.npy")
        params = flax.core.unfreeze(params)
        params['embed_tokens']['embedding'] = jnp.array(embeds)
        params = flax.core.freeze(params)

    # Cosine decay LR + warmup via optax chain
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=3e-4,
        warmup_steps=500,
        decay_steps=50_000,
        end_value=3e-5,
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
        optax.clip_by_global_norm(1.0),   # gradient clipping
        base_tx,
    )
    # Wrap dengan MultiSteps untuk gradient accumulation
    tx = optax.MultiSteps(tx, every_k_schedule=GRAD_ACCUM_STEPS)

    state = MAMoETrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, memory_state

# ── Training Step (pmap) ─────────────────────────────────────────────────────
@functools.partial(jax.pmap, axis_name='batch')
def train_step(state, memory_state, batch_enc_inputs, batch_dec_inputs):
    labels = jnp.roll(batch_dec_inputs, shift=-1, axis=1).at[:, -1].set(0)

    def loss_fn(params):
        (logits, _, _, aux_loss, avg_f_i), mutated = state.apply_fn(
            {'params': params, 'memory': memory_state},
            input_ids=batch_enc_inputs,
            decoder_input_ids=batch_dec_inputs,
            mutable=['memory'],
        )
        vocab_size   = logits.shape[-1]
        log_probs    = jax.nn.log_softmax(logits, axis=-1)
        ce_loss    = -jnp.sum(jax.nn.one_hot(labels, vocab_size) * log_probs, axis=-1)
        loss_mask  = (labels != 0).astype(jnp.float32)
        mean_ce    = jnp.sum(ce_loss * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)
        total_loss   = mean_ce + 0.01 * aux_loss
        return total_loss, (mean_ce, aux_loss, avg_f_i, mutated.get('memory', {}))

    (total_loss, (ce_loss, aux_loss, avg_f_i, new_mem)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)

    grads      = jax.lax.pmean(grads,      axis_name='batch')
    total_loss = jax.lax.pmean(total_loss, axis_name='batch')
    ce_loss    = jax.lax.pmean(ce_loss,    axis_name='batch')
    aux_loss   = jax.lax.pmean(aux_loss,   axis_name='batch')
    avg_f_i    = jax.lax.pmean(avg_f_i,    axis_name='batch')

    state = state.apply_gradients(grads=grads)
    return state, new_mem, {'loss': total_loss, 'ce_loss': ce_loss, 'aux_loss': aux_loss, 'expert_load': avg_f_i}

# ── Data Loading ─────────────────────────────────────────────────────────────
def resolve_data_paths():
    """Pilih path dataset: Prioritaskan .npy agar RAM cepat."""
    if os.path.exists(KAGGLE_TOKENS):
        print(f"⚡ Fast path: loading pre-tokenized npy → {KAGGLE_TOKENS}")
        return 'npy', KAGGLE_TOKENS, None
    if os.path.exists(LOCAL_TOKENS):
        print(f"⚡ Fast path: loading pre-tokenized npy → {LOCAL_TOKENS}")
        return 'npy', LOCAL_TOKENS, None
    if os.path.exists(KAGGLE_CSV):
        print(f"⚠️  npy tidak ditemukan, streaming CSV → {KAGGLE_CSV}")
        return 'csv', KAGGLE_CSV, KAGGLE_TOK
    if os.path.exists(LOCAL_CSV):
        print(f"⚠️  npy tidak ditemukan, streaming CSV → {LOCAL_CSV}")
        return 'csv', LOCAL_CSV, LOCAL_TOK
    if os.path.exists(COLAB_CSV):
        print(f"✅ Streaming File (dengan Regex Cleaning) → {COLAB_CSV}")
        return 'csv', COLAB_CSV, LOCAL_TOK
    raise FileNotFoundError("Tidak ada dataset mentah ditemukan!")

def npy_epoch_generator(npy_path, total_batch_size, seq_len):
    """Load npy PENUH ke RAM (bukan mmap), yield batch acak — MAKSIMAL CEPAT."""
    print("Loading pre-tokenized dataset...")
    tokens = np.load(npy_path)
    
    if os.environ.get('QUICK_TEST') == '1':
        print("\n⚠️ QUICK_TEST MODE ENABLED! Truncating data to 10 steps...")
        tokens = tokens[: total_batch_size * SEQ_LEN * 10]
        
    total_tokens_in_file = len(tokens)                   # full RAM load, no disk I/O during training
    total  = len(tokens)
    usable = (total // seq_len) * seq_len
    arr    = tokens[:usable].reshape(-1, seq_len)
    print(f"   Dataset: {arr.shape[0]:,} sequences × {seq_len} tokens ({arr.nbytes/1e6:.0f} MB in RAM)")
    idx = np.random.permutation(len(arr))
    for i in range(0, len(idx) - total_batch_size, total_batch_size):
        yield arr[idx[i : i + total_batch_size]]

def csv_epoch_generator(csv_path, tok_path, total_batch_size, seq_len):
    """Fallback: streaming + tokenize on-the-fly (lebih lambat)."""
    import pandas as pd
    from tokenizers import Tokenizer
    tokenizer  = Tokenizer.from_file(tok_path)
    for chunk in pd.read_csv(csv_path, chunksize=50000):
        col   = next((c for c in ["text","prompt","content","completion","text_clean","article"]
                      if c in chunk.columns), chunk.columns[0])
        texts = chunk[col].dropna().astype(str).tolist()
        
        ids   = []
        for enc in tokenizer.encode_batch(texts):
            ids.extend(enc.ids)
        n   = len(ids) // seq_len
        arr = np.array(ids[:n * seq_len], dtype=np.uint16).reshape(n, seq_len)
        np.random.shuffle(arr)
        for i in range(0, n - total_batch_size, total_batch_size):
            yield arr[i : i + total_batch_size]

def prefetch(generator, maxsize=PREFETCH_QUEUE):
    """Jalankan data generator di background thread, buffer ke queue."""
    q = queue.Queue(maxsize=maxsize)
    sentinel = object()

    def worker():
        for item in generator:
            q.put(item)
        q.put(sentinel)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Colab TPU Initialization
    if 'COLAB_TPU_ADDR' in os.environ:
        import jax.tools.colab_tpu as colab_tpu
        colab_tpu.setup_tpu()
        print("✅ Colab TPU initialized.")

    num_devices      = jax.device_count()
    total_batch_size = LOCAL_BATCH_SIZE * num_devices
    print(f"Devices        : {num_devices}")
    print(f"Micro-batch    : {total_batch_size} total ({LOCAL_BATCH_SIZE} per device)")
    print(f"Grad accum     : {GRAD_ACCUM_STEPS} steps")
    print(f"Effective batch: {total_batch_size * GRAD_ACCUM_STEPS} (Safe for 16GB VRAM)")
    print(f"Seq len        : {SEQ_LEN}")
    print()



    config = MAMoEConfig()
    model  = MAMoEForConditionalGeneration(config=config)
    rng    = jax.random.PRNGKey(42)

    dummy_enc = jnp.ones((LOCAL_BATCH_SIZE, ENC_SEQ_LEN), dtype=jnp.int32)
    dummy_dec = jnp.ones((LOCAL_BATCH_SIZE, DEC_SEQ_LEN), dtype=jnp.int32)
    print("Initializing model weights & optimizer...")
    state, memory_state = create_train_state(rng, model, dummy_enc, dummy_dec)

    state        = replicate(state)
    memory_state = replicate(memory_state)
    print("Done.\n")
    # Catatan: Memory TIDAK di-reset selama pre-training.
    # Memory dibiarkan mengakumulasi konteks dari teks secara natural.

    print(f"Starting Phase 1: Full Pre-Training ({NUM_EPOCHS} Epochs) ...\n")
    start_time     = time.time()
    last_log_time  = start_time
    total_tokens   = 0
    accum_grads    = None

    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== EPOCH {epoch}/{NUM_EPOCHS} ==========")
        # Pilih sumber data
        mode, data_path, tok_path = resolve_data_paths()
        if mode == 'npy':
            raw_gen = npy_epoch_generator(data_path, total_batch_size, SEQ_LEN)
        else:
            raw_gen = csv_epoch_generator(data_path, tok_path, total_batch_size, SEQ_LEN)
    
        dataloader = prefetch(raw_gen)   # background prefetch
        
        for batch in dataloader:
            global_step += 1
            # Cast to int32 for JAX
            batch         = batch.astype(np.int32)
            
            batch_enc = batch[:, :ENC_SEQ_LEN]
            batch_dec = batch[:, ENC_SEQ_LEN:]
            
            sharded_enc = batch_enc.reshape((num_devices, LOCAL_BATCH_SIZE, ENC_SEQ_LEN))
            sharded_dec = batch_dec.reshape((num_devices, LOCAL_BATCH_SIZE, DEC_SEQ_LEN))
    
            state, memory_state, metrics = train_step(state, memory_state, sharded_enc, sharded_dec)
            total_tokens += total_batch_size * SEQ_LEN
    
            if global_step % LOG_INTERVAL == 0:
                now           = time.time()
                tok_per_sec   = (total_batch_size * SEQ_LEN * LOG_INTERVAL) / (now - last_log_time)
                elapsed       = int(now - start_time)
                ce_val        = float(unreplicate(metrics['ce_loss']))
                aux_val       = float(unreplicate(metrics['aux_loss']))
                expert_load   = np.array(unreplicate(metrics['expert_load']))
    
                # Format expert load: E0:12% E1:8% ...
                expert_str = ' '.join(
                    f'E{i}:{v*100:.0f}%' for i, v in enumerate(expert_load)
                )
                print(
                    f"Epoch {epoch} | Step {global_step:06d} | "
                    f"CE {ce_val:.4f} | "
                    f"Aux {aux_val:.4f} | "
                    f"Speed {tok_per_sec:>8,.0f} tok/s | "
                    f"Elapsed {elapsed//60}m {elapsed%60:02d}s"
                )
                print(f"          Expert load: [{expert_str}]")
                last_log_time = now

        # --- SAVE CHECKPOINT PYTORCH PER EPOCH ---
        print(f"\n💾 Saving PyTorch checkpoint for Epoch {epoch}...")
        ckpt_dir = '/kaggle/working/checkpoints' if os.path.exists('/kaggle') else 'checkpoints'
        os.makedirs(ckpt_dir, exist_ok=True)
        
        unreplicated_state = unreplicate(state)
        
        config_pt = PyTorchConfig()
        pt_model = PyTorchMAMoE(config_pt)
        state_dict = convert_jax_to_pytorch(unreplicated_state.params, pt_model, config_pt)
        
        out_path = os.path.join(ckpt_dir, f'pytorch_model_epoch_{epoch}.pt')
        torch.save(state_dict, out_path)
        print(f"✅ PyTorch Checkpoint saved to: {out_path}\n")

    total_elapsed = int(time.time() - start_time)
    print(f"\n✅ Phase 1 Complete!  Total: {total_elapsed//3600}h {(total_elapsed%3600)//60}m  |  Tokens: {total_tokens:,}")

if __name__ == '__main__':
    main()
