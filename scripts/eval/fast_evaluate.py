import os
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
, sys, json, time
import numpy as np
import jax
import jax.numpy as jnp
from tokenizers import Tokenizer
import orbax.checkpoint as ocp

from mamoe.config import MAMoEConfig
from mamoe.model import MAMoEForCausalLM

LOCAL_TOK = 'tokenizer_hf/tokenizer.json'
KAGGLE_TOK = '/kaggle/input/datasets/akhyarsafrudin/tokenizer_hf/tokenizer.json'

def load_model_and_tokenizer():
    devices = jax.devices()
    print(f"🖥️  JAX Devices: {devices}")
    
    tok_path = KAGGLE_TOK if os.path.exists(KAGGLE_TOK) else LOCAL_TOK
    tokenizer = Tokenizer.from_file(tok_path)

    config = MAMoEConfig()
    model  = MAMoEForCausalLM(config=config)
    rng    = jax.random.PRNGKey(0)

    dummy = jnp.ones((1, 64), dtype=jnp.int32)
    variables = model.init(rng, dummy)
    
    ckpt_dir = '/kaggle/working/checkpoints/phase2'
    if not os.path.exists(ckpt_dir):
        ckpt_dir = '/kaggle/working/checkpoints/phase1'
        
    if os.path.exists(ckpt_dir):
        print(f"Loading checkpoint from {ckpt_dir}...")
        checkpointer = ocp.StandardCheckpointer()
        raw_state = checkpointer.restore(os.path.abspath(ckpt_dir), target=None)
        if 'params' in raw_state: params = raw_state['params']
        else: params = raw_state
        variables = {'params': params, 'memory': variables.get('memory', {})}
        print("✅ Checkpoint Loaded!")
    else:
        print("⚠️ Warning: No checkpoint found. Using random weights.")
    
    return model, variables, tokenizer

def build_prompt(episode):
    turns = []
    for turn in episode['conversation']:
        role = "User" if turn['role'] == 'user' else "Assistant"
        turns.append(f"{role}: {turn['text']}")
    turns.append(f"User: {episode['query']}")
    turns.append("Assistant:")
    return "\n".join(turns)

def main():
    model, variables, tokenizer = load_model_and_tokenizer()
    
    jsonl_path = '/kaggle/input/datasets/akhyarsafrudin/dataset-chat/memorybench_train_samples.jsonl'
    if not os.path.exists(jsonl_path):
        jsonl_path = 'data/memorybench_train_samples.jsonl'
        if not os.path.exists(jsonl_path):
            print("⚠️ Dataset not found!")
            return
            
    episodes = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip(): episodes.append(json.loads(line.strip()))
            
    print(f"\n🚀 Menjalankan Fast Teacher-Forcing Evaluation pada {len(episodes)} episodes...")
    
    # JIT the forward pass for a fixed batch size and sequence length
    BATCH_SIZE = 32
    SEQ_LEN = 512
    
    @jax.jit
    def forward_batch(tokens_in, mask_in):
        mem = variables.get('memory', {})
        # Mask shape: (batch, 1, 1, seq_len)
        mask_in = mask_in.reshape((BATCH_SIZE, 1, 1, SEQ_LEN))
        (logits, _, _, _, _), _ = model.apply(
            {'params': variables['params'], 'memory': mem},
            tokens_in,
            attention_mask=mask_in,
            mutable=['memory']
        )
        return logits

    correct = 0
    total = 0
    t0 = time.time()
    
    for i in range(0, len(episodes), BATCH_SIZE):
        batch_eps = episodes[i:i+BATCH_SIZE]
        actual_bs = len(batch_eps)
        
        batch_tokens = np.zeros((BATCH_SIZE, SEQ_LEN), dtype=np.int32)
        batch_mask = np.full((BATCH_SIZE, SEQ_LEN), -1e9, dtype=np.float32)
        answer_spans = []
        
        for j, ep in enumerate(batch_eps):
            prompt = build_prompt(ep)
            prompt_ids = tokenizer.encode(prompt).ids
            ans_ids = tokenizer.encode(" " + ep['answer']).ids
            
            # Combine prompt and answer
            full_ids = (prompt_ids + ans_ids)[-SEQ_LEN:]
            
            # Record where the answer tokens are
            ans_len = len(ans_ids)
            prompt_len = len(full_ids) - ans_len
            answer_spans.append((prompt_len, len(full_ids), ans_ids))
            
            batch_tokens[j, :len(full_ids)] = full_ids
            batch_mask[j, :len(full_ids)] = 0.0
            
        # GPU Forward pass (Semua tebakan dalam sekali jalan)
        logits = forward_batch(batch_tokens, batch_mask)
        logits = jax.device_get(logits) # Bawa ke CPU
        
        # Cocokkan jawaban dengan CPU
        for j in range(actual_bs):
            p_start, p_end, ans_ids = answer_spans[j]
            if p_start < 0: continue
            
            # The model predicts token at t based on t-1
            # So the logits for the answer are at indices [p_start-1 : p_end-1]
            pred_logits = logits[j, p_start-1 : p_end-1]
            pred_ids = np.argmax(pred_logits, axis=-1).tolist()
            
            if pred_ids == ans_ids:
                correct += 1
            total += 1
            
        if (i + BATCH_SIZE) % 128 == 0 or (i + actual_bs) == len(episodes):
            print(f"  [{i+actual_bs:4d}/{len(episodes)}] Accuracy (Exact Match): {correct/total*100:.2f}% | {time.time()-t0:.1f}s elapsed")

    print(f"\n✅ Evaluasi Super Cepat Selesai!")
    print(f"   Total Waktu: {time.time()-t0:.1f} detik")
    print(f"   Akurasi (Exact Match): {correct/total*100:.2f}%")

if __name__ == "__main__":
    main()
