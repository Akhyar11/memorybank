import os
import time
import threading
import queue
import jax
import jax.numpy as jnp
import optax
import numpy as np
import functools

# ── Optimasi Tensor Core (T4/P100 bfloat16) ─────────────────────────────────
jax.config.update('jax_default_matmul_precision', 'bfloat16')

from flax.training import train_state
from flax.jax_utils import replicate, unreplicate

from mamoe.config import MAMoEConfig
from mamoe.model import MAMoEForCausalLM

# ─── Path Konfigurasi ────────────────────────────────────────────────────────
KAGGLE_TOKENS   = '/kaggle/working/vqfat_tokens.npy'
KAGGLE_CSV      = '/kaggle/input/datasets/akhyarsafrudin/dataset-chat/vqfat_cosmopedia_id.csv'
KAGGLE_TOK      = '/kaggle/input/datasets/akhyarsafrudin/tokenizer/tokenizer.json'
LOCAL_TOKENS    = 'data/pretrain/vqfat_tokens.npy'
LOCAL_CSV       = 'data/raw/vqfat_cosmopedia_id.csv'
LOCAL_TOK       = 'tokenizer/tokenizer.json'
# ─────────────────────────────────────────────────────────────────────────────

# ─── Hyperparameters ─────────────────────────────────────────────────────────
SEQ_LEN           = 1024
LOCAL_BATCH_SIZE  = 4       # per device  → Total = 4 × num_devices
RESET_INTERVAL    = 4       # memory reset setiap N step
LOG_INTERVAL      = 10
GRAD_ACCUM_STEPS  = 2       # gradient accumulation (effective batch × 2)
PREFETCH_QUEUE    = 8       # buffer batches di RAM sebelum GPU butuh
# ─────────────────────────────────────────────────────────────────────────────

class MAMoETrainState(train_state.TrainState):
    pass

# ── Model Init ───────────────────────────────────────────────────────────────
def create_train_state(rng, model, dummy_input):
    variables      = model.init(rng, dummy_input)
    params         = variables['params']
    memory_state   = variables.get('memory', {})

    # Cosine decay LR + warmup via optax chain
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=3e-4,
        warmup_steps=500,
        decay_steps=50_000,
        end_value=3e-5,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),   # gradient clipping
        optax.adamw(lr_schedule, weight_decay=0.1),
    )

    state = MAMoETrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, memory_state

def get_empty_memory_state(rng, model, dummy_input):
    variables = model.init(rng, dummy_input)
    return variables.get('memory', {})

# ── Training Step (pmap) ─────────────────────────────────────────────────────
@functools.partial(jax.pmap, axis_name='batch')
def train_step(state, memory_state, batch_inputs):
    labels = jnp.roll(batch_inputs, shift=-1, axis=1).at[:, -1].set(0)

    def loss_fn(params):
        (logits, _, _, aux_loss), mutated = state.apply_fn(
            {'params': params, 'memory': memory_state},
            batch_inputs,
            mutable=['memory'],
        )
        vocab_size   = logits.shape[-1]
        log_probs    = jax.nn.log_softmax(logits, axis=-1)
        ce_loss      = -jnp.sum(jax.nn.one_hot(labels, vocab_size) * log_probs, axis=-1)
        mean_ce      = jnp.mean(ce_loss)
        total_loss   = mean_ce + 0.1 * aux_loss
        return total_loss, (mean_ce, aux_loss, mutated.get('memory', {}))

    (total_loss, (ce_loss, aux_loss, new_mem)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)

    grads      = jax.lax.pmean(grads,      axis_name='batch')
    total_loss = jax.lax.pmean(total_loss, axis_name='batch')
    ce_loss    = jax.lax.pmean(ce_loss,    axis_name='batch')
    aux_loss   = jax.lax.pmean(aux_loss,   axis_name='batch')

    state = state.apply_gradients(grads=grads)
    return state, new_mem, {'loss': total_loss, 'ce_loss': ce_loss, 'aux_loss': aux_loss}

# ── Data Loading ─────────────────────────────────────────────────────────────
def resolve_data_paths():
    """Pilih path dataset: npy → csv kaggle → csv lokal"""
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
    raise FileNotFoundError("Tidak ada dataset ditemukan!")

def npy_epoch_generator(npy_path, total_batch_size, seq_len):
    """Load npy PENUH ke RAM (bukan mmap), yield batch acak — MAKSIMAL CEPAT."""
    print(f"   Loading npy fully into RAM...")
    tokens = np.load(npy_path)                         # full RAM load, no disk I/O during training
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
    num_devices      = jax.device_count()
    total_batch_size = LOCAL_BATCH_SIZE * num_devices
    print(f"Devices   : {num_devices}")
    print(f"Batch size: {total_batch_size} total ({LOCAL_BATCH_SIZE} per device)")
    print(f"Seq len   : {SEQ_LEN}")
    print(f"Grad accum: {GRAD_ACCUM_STEPS}  →  Effective batch: {total_batch_size * GRAD_ACCUM_STEPS}")
    print()

    config = MAMoEConfig()
    model  = MAMoEForCausalLM(config=config)
    rng    = jax.random.PRNGKey(42)

    dummy  = jnp.ones((LOCAL_BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
    print("Initializing model weights & optimizer...")
    state, memory_state = create_train_state(rng, model, dummy)

    # Pre-compute empty memory template ONCE
    empty_memory_template = jax.tree_util.tree_map(jnp.zeros_like, memory_state)

    state        = replicate(state)
    memory_state = replicate(memory_state)
    print("Done.\n")

    # Pilih sumber data
    mode, data_path, tok_path = resolve_data_paths()
    if mode == 'npy':
        raw_gen = npy_epoch_generator(data_path, total_batch_size, SEQ_LEN)
    else:
        raw_gen = csv_epoch_generator(data_path, tok_path, total_batch_size, SEQ_LEN)

    dataloader = prefetch(raw_gen)   # background prefetch

    print("Starting Phase 1: Full Pre-Training (1 Epoch) ...\n")
    start_time     = time.time()
    last_log_time  = start_time
    total_tokens   = 0
    accum_grads    = None

    for step, batch in enumerate(dataloader, 1):
        # Cast to int32 for JAX
        batch         = batch.astype(np.int32)
        sharded_batch = batch.reshape((num_devices, LOCAL_BATCH_SIZE, SEQ_LEN))

        state, memory_state, metrics = train_step(state, memory_state, sharded_batch)
        total_tokens += total_batch_size * SEQ_LEN

        if step % LOG_INTERVAL == 0:
            now            = time.time()
            tok_per_sec    = (total_batch_size * SEQ_LEN * LOG_INTERVAL) / (now - last_log_time)
            elapsed        = int(now - start_time)
            ce_val         = float(unreplicate(metrics['ce_loss']))
            aux_val        = float(unreplicate(metrics['aux_loss']))
            print(
                f"Step {step:06d} | "
                f"CE {ce_val:.4f} | "
                f"Aux {aux_val:.4f} | "
                f"Speed {tok_per_sec:>8,.0f} tok/s | "
                f"Elapsed {elapsed//60}m {elapsed%60:02d}s"
            )
            last_log_time = now

        if step % RESET_INTERVAL == 0:
            # Nol-kan memory_state yang sedang berjalan (sharding identik, tidak re-trace)
            memory_state = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), memory_state)

    total_elapsed = int(time.time() - start_time)
    print(f"\n✅ Phase 1 Complete!  Total: {total_elapsed//3600}h {(total_elapsed%3600)//60}m  |  Tokens: {total_tokens:,}")

if __name__ == '__main__':
    main()
