"""
evaluate.py
===========
Pipeline evaluasi MAMoE-50 sesuai Research Questions di map.md.

Metrik yang dihitung:
  - Memory Recall Accuracy (MRA)          ← RQ1, RQ2
  - Answer Accuracy (EM & Partial)        ← RQ2
  - Memory Update Accuracy                ← RQ2
  - Context Compression Ratio (CCR)       ← RQ3
  - Long-range Recall (@ turns distance)  ← H1, H2
  - Expert Load Distribution (Aux)        ← Health check MoE

Jalankan: python evaluate.py
"""

import os, sys, json, re, time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

# ── Optimasi ───────────────────────────────────────────────────────────────
jax.config.update('jax_default_matmul_precision', 'bfloat16')

from tokenizers import Tokenizer

from mamoe.config import MAMoEConfig
from mamoe.model import MAMoEForCausalLM

# ─── Path Konfigurasi ───────────────────────────────────────────────────────
KAGGLE_TOK   = '/kaggle/input/datasets/akhyarsafrudin/tokenizer_hf/tokenizer.json'
LOCAL_TOK    = 'tokenizer_hf/tokenizer.json'

# Dataset evaluasi (gunakan memorybench_train.jsonl sebagai held-out eval)
EVAL_JSONL   = 'data/memorybench_train.jsonl'
SAMPLES_JSONL= 'data/memorybench_train_samples.jsonl'

MAX_NEW_TOKENS = 30
TEMPERATURE    = 0.0   # greedy decoding untuk evaluasi deterministik
# ────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
#  1. Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    devices = jax.devices()
    print(f"🖥️  JAX Devices: {devices}")
    if devices[0].platform == 'cpu':
        print("❌ ERROR: JAX IS RUNNING ON CPU! Please run '!pip install -q -U \"jax[cuda12]\"' first!")
        
    tok_path = KAGGLE_TOK if os.path.exists(KAGGLE_TOK) else LOCAL_TOK
    tokenizer = Tokenizer.from_file(tok_path)

    config = MAMoEConfig()
    model  = MAMoEForCausalLM(config=config)
    rng    = jax.random.PRNGKey(0)

    # Init params
    dummy = jnp.ones((1, 64), dtype=jnp.int32)
    variables = model.init(rng, dummy)
    
    # Load from checkpoint
    ckpt_dir = '/kaggle/working/checkpoints/phase2'
    if not os.path.exists(ckpt_dir):
        ckpt_dir = '/kaggle/working/checkpoints/phase1'
        
    if os.path.exists(ckpt_dir):
        print(f"   Loading checkpoint from {ckpt_dir}...")
        checkpointer = ocp.StandardCheckpointer()
        
        raw_state = checkpointer.restore(os.path.abspath(ckpt_dir), target=None)
        
        # Ekstrak params dari TrainState (yang memiliki 'step', 'params', 'opt_state')
        if 'params' in raw_state:
            params = raw_state['params']
        else:
            params = raw_state
            
        variables = {'params': params, 'memory': variables.get('memory', {})}
            
        print("✅ Checkpoint Loaded successfully!")
    else:
        print("⚠️ WARNING: No checkpoint found! Evaluating with RANDOM weights!")

    print(f"✅ Model initialized (evaluation mode)")
    print(f"   Tokenizer: {tok_path}")
    return model, variables, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
#  2. Inference
# ══════════════════════════════════════════════════════════════════════════════

def greedy_generate_batch(model, variables, tokenizer, prompts: List[str],
                          max_new_tokens: int = MAX_NEW_TOKENS) -> List[str]:
    """Auto-regressive greedy decoding dengan Memory Bank (Batched + JIT)."""
    batch_size = len(prompts)
    encs = tokenizer.encode_batch(prompts)
    
    # Pad to fixed length
    MAX_PROMPT_LEN = 256
    MAX_LEN = MAX_PROMPT_LEN + max_new_tokens
    
    padded_ids = np.zeros((batch_size, MAX_LEN), dtype=np.int32)
    seq_lens = np.zeros(batch_size, dtype=np.int32)
    
    for i, enc in enumerate(encs):
        ids = enc.ids[-MAX_PROMPT_LEN:]
        seq_lens[i] = len(ids)
        padded_ids[i, :len(ids)] = ids
        
    tokens = jnp.array(padded_ids, dtype=jnp.int32)
    seq_lens_jnp = jnp.array(seq_lens, dtype=jnp.int32)
    
    # Duplicate memory state for the batch
    mem = variables.get('memory', {})
    if len(jax.tree_util.tree_leaves(mem)) > 0:
        mem = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, batch_size, axis=0) if (jnp.ndim(x) > 0 and x.shape[0] == 1) else (x[:batch_size] if jnp.ndim(x) > 0 else x), 
            mem
        )
    
    params = variables['params']

    @jax.jit
    def generate_step(tokens_in, seq_lens_in, mem_in):
        # Create additive attention mask
        idx = jnp.arange(MAX_LEN)[None, :]
        mask = idx < seq_lens_in[:, None]
        mask = jnp.where(mask, 0.0, -1e9)
        mask = mask.reshape((batch_size, 1, 1, MAX_LEN))
        
        (logits, _, _, _, _), mutated = model.apply(
            {'params': params, 'memory': mem_in},
            tokens_in,
            attention_mask=mask,
            mutable=['memory'],
        )
        
        batch_idx = jnp.arange(batch_size)
        last_logits = logits[batch_idx, seq_lens_in - 1]
        next_toks = jnp.argmax(last_logits, axis=-1).astype(jnp.int32)
        
        tokens_out = tokens_in.at[batch_idx, seq_lens_in].set(next_toks)
        mem_out = mutated.get('memory', mem_in)
        return tokens_out, next_toks, mem_out

    # Generation loop
    active = np.ones(batch_size, dtype=bool)
    for step in range(max_new_tokens):
        tokens, next_toks, mem = generate_step(tokens, seq_lens_jnp, mem)
        next_toks = np.array(next_toks)
        seq_lens_jnp += 1
        
        for i in range(batch_size):
            if active[i]:
                dec = tokenizer.decode([int(next_toks[i])])
                if dec in ('\n', '</s>', '[EOS]', '<eos>'):
                    active[i] = False
                    
        if not np.any(active):
            break

    generated = []
    tokens_np = np.array(tokens)
    seq_lens_np = np.array(seq_lens_jnp)
    for i in range(batch_size):
        start_idx = seq_lens[i]
        end_idx = seq_lens_np[i]
        gen = tokenizer.decode(tokens_np[i, start_idx:end_idx].tolist())
        generated.append(gen.strip())
        
    return generated


# ══════════════════════════════════════════════════════════════════════════════
#  3. Metrik
# ══════════════════════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    """Lowercase, strip punctuation."""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)

def partial_match(pred: str, gold: str) -> bool:
    """Benar jika gold ada di dalam prediksi (substring)."""
    return normalize(gold) in normalize(pred)

@dataclass
class EvalResult:
    task:              str
    episode_id:        str
    query:             str
    gold_answer:       str
    prediction:        str
    em:                bool
    partial:           bool
    memory_distance:   int
    num_facts:         int
    num_updates:       int
    context_tokens:    int          # jumlah token conversation history
    memory_slots_used: int = 0      # berapa slot memory yang di-write


# ══════════════════════════════════════════════════════════════════════════════
#  4. Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

def load_episodes(jsonl_path: str) -> List[Dict]:
    episodes = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes

def build_prompt(episode: Dict, tokenizer: Tokenizer) -> tuple:
    """
    Bangun prompt dari conversation history + query.
    Return (prompt_str, context_token_count).
    """
    turns = []
    for turn in episode['conversation']:
        role = "User" if turn['role'] == 'user' else "Assistant"
        turns.append(f"{role}: {turn['text']}")

    turns.append(f"User: {episode['query']}")
    turns.append("Assistant:")

    prompt = "\n".join(turns)
    tok_count = len(tokenizer.encode(prompt).ids)
    return prompt, tok_count


# ══════════════════════════════════════════════════════════════════════════════
#  5. Evaluasi utama
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(model, variables, tokenizer,
                   episodes: List[Dict],
                   max_episodes: Optional[int] = None,
                   label: str = "eval",
                   batch_size: int = 8) -> List[EvalResult]:
    results = []
    n = min(len(episodes), max_episodes) if max_episodes else len(episodes)
    episodes = episodes[:n]
    print(f"\n📊 Running {label} on {n} episodes (Batched, batch_size={batch_size})...")

    t0 = time.time()
    
    for i in range(0, n, batch_size):
        batch_eps = episodes[i:i+batch_size]
        actual_bs = len(batch_eps)
        
        prompts = []
        ctx_tokens_list = []
        for ep in batch_eps:
            prompt, ctx_tokens = build_prompt(ep, tokenizer)
            prompts.append(prompt)
            ctx_tokens_list.append(ctx_tokens)
            
        # Jika batch_size di sisa data tidak penuh, kita tetap kirim semuanya ke greedy_generate_batch
        # (JIT akan re-compile 1 kali lagi untuk ukuran batch sisa ini, tapi tidak masalah)
        preds = greedy_generate_batch(model, variables, tokenizer, prompts)
        
        for j, ep in enumerate(batch_eps):
            gold = ep['answer']
            diff = ep['difficulty']
            pred = preds[j]
            r = EvalResult(
                task              = ep['task'],
                episode_id        = ep['episode_id'],
                query             = ep['query'],
                gold_answer       = gold,
                prediction        = pred,
                em                = exact_match(pred, gold),
                partial           = partial_match(pred, gold),
                memory_distance   = diff.get('memory_distance', 0),
                num_facts         = diff.get('num_facts', 0),
                num_updates       = diff.get('num_updates', 0),
                context_tokens    = ctx_tokens_list[j],
                memory_slots_used = diff.get('num_facts', 0),
            )
            results.append(r)

        # Progress
        mra = sum(x.partial for x in results) / len(results)
        elapsed = time.time() - t0
        print(f"  [{i+actual_bs:4d}/{n}] MRA(partial): {mra:.3f} | {elapsed:.0f}s elapsed")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  6. Agregasi & Laporan
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: List[EvalResult]) -> Dict:
    n = len(results)
    if n == 0:
        return {}

    em_acc        = sum(r.em for r in results) / n
    partial_acc   = sum(r.partial for r in results) / n

    # --- Memory Recall Accuracy sesuai map.md ---
    mra = partial_acc   # MRA = correct_retrieved / total_queries

    # --- Context Compression Ratio ---
    # CCR = avg_context_tokens / avg_slots_used
    avg_ctx  = np.mean([r.context_tokens for r in results])
    avg_mem  = np.mean([r.memory_slots_used for r in results])
    ccr      = avg_ctx / max(avg_mem, 1)

    # --- Long-range Recall (by memory distance buckets) ---
    buckets = defaultdict(list)
    for r in results:
        d = r.memory_distance
        if d <= 2:
            buckets['≤2 turns'].append(r.partial)
        elif d <= 5:
            buckets['3-5 turns'].append(r.partial)
        elif d <= 10:
            buckets['6-10 turns'].append(r.partial)
        elif d <= 20:
            buckets['11-20 turns'].append(r.partial)
        else:
            buckets['>20 turns'].append(r.partial)

    long_range = {k: (np.mean(v), len(v)) for k, v in sorted(buckets.items())}

    # --- Memory Update Accuracy ---
    update_eps = [r for r in results if r.num_updates > 0]
    update_acc = sum(r.partial for r in update_eps) / len(update_eps) if update_eps else None

    return {
        'n_episodes'      : n,
        'exact_match'     : em_acc,
        'partial_match'   : partial_acc,
        'MRA'             : mra,
        'CCR'             : ccr,
        'avg_ctx_tokens'  : avg_ctx,
        'avg_mem_slots'   : avg_mem,
        'long_range_recall': long_range,
        'memory_update_acc': update_acc,
    }


def print_report(metrics: Dict, label: str = "Evaluation Report"):
    bar = "═" * 55
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"  Episodes evaluated   : {metrics['n_episodes']}")
    print(f"  Exact Match (EM)     : {metrics['exact_match']:.1%}")
    print(f"  Partial Match        : {metrics['partial_match']:.1%}")
    print()
    print(f"  ── Memory Metrics (sesuai map.md §16) ──────────────")
    print(f"  Memory Recall Acc    : {metrics['MRA']:.1%}")
    if metrics['memory_update_acc'] is not None:
        print(f"  Memory Update Acc    : {metrics['memory_update_acc']:.1%}")
    else:
        print(f"  Memory Update Acc    : N/A (no update episodes)")
    print(f"  Context Compression  : {metrics['CCR']:.1f}×")
    print(f"  Avg ctx tokens       : {metrics['avg_ctx_tokens']:.0f}")
    print(f"  Avg memory slots     : {metrics['avg_mem_slots']:.1f}")
    print()
    print(f"  ── Long-range Recall (by memory distance) ──────────")
    for dist, (acc, cnt) in metrics['long_range_recall'].items():
        bar_len = int(acc * 30)
        bar_str = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {dist:>12} | {bar_str} {acc:.1%} (n={cnt})")
    print(f"\n{'═'*55}\n")


def save_report(metrics: Dict, results: List[EvalResult], out_path: str = "eval_report.json"):
    report = {
        'metrics': metrics,
        'per_episode': [
            {
                'episode_id':      r.episode_id,
                'task':            r.task,
                'query':           r.query,
                'gold':            r.gold_answer,
                'pred':            r.prediction,
                'em':              r.em,
                'partial':         r.partial,
                'memory_distance': r.memory_distance,
                'num_facts':       r.num_facts,
                'num_updates':     r.num_updates,
            }
            for r in results
        ]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"💾 Report saved to: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  7. Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  MAMoE-50 Evaluation Pipeline")
    print("  (sesuai Research Questions di map.md)")
    print("=" * 55)

    model, variables, tokenizer = load_model_and_tokenizer()

    # Pilih dataset
    eval_kaggle = '/kaggle/input/datasets/akhyarsafrudin/dataset-chat/memorybench_train_samples.jsonl'
    eval_local  = 'data/memorybench_train.jsonl'
    eval_path   = eval_kaggle if os.path.exists(eval_kaggle) else eval_local

    if os.path.exists(eval_path):
        episodes = load_episodes(eval_path)
        print(f"📂 Dataset: {eval_path} ({len(episodes)} episodes)")
    else:
        print(f"⚠️ Eval dataset tidak ditemukan di {eval_path}. Fase evaluasi dilewati.")
        print("💡 Catatan: Harap upload memorybench_train.jsonl ke Kaggle Dataset (dataset-chat) jika Anda ingin evaluasi berjalan.")
        sys.exit(0)
    # Batasi jumlah episode untuk evaluasi cepat (hapus batas untuk full eval)
    MAX_EVAL = None   # None = evaluasi semua

    results = run_evaluation(
        model, variables, tokenizer,
        episodes,
        max_episodes=MAX_EVAL,
        label="MemoryBench ID"
    )

    metrics = compute_metrics(results)
    print_report(metrics, label="MAMoE-50 Memory Evaluation Report")

    out = '/kaggle/working/eval_report.json' if os.path.exists('/kaggle') else 'eval_report.json'
    save_report(metrics, results, out_path=out)


if __name__ == '__main__':
    main()
