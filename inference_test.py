import os

import jax
import jax.numpy as jnp
from tokenizers import Tokenizer
import time

from mamoe.config import MAMoEConfig
from mamoe.model import MAMoEForCausalLM

def main():
    print("1. Memuat Tokenizer...")
    tokenizer = Tokenizer.from_file('tokenizer_hf/tokenizer.json')
    
    print("2. Inisialisasi Model MAMoE-50 (Bobot Random Lokal)...")
    config = MAMoEConfig()
    model = MAMoEForCausalLM(config=config)
    rng = jax.random.PRNGKey(42)
    
    # Hanya batch size = 1 untuk inference
    dummy = jnp.ones((1, 32), dtype=jnp.int32)
    variables = model.init(rng, dummy)
    
    # Hitung ukuran model di RAM
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables))
    print(f"   Total Parameter: {param_count:,}")
    print(f"   Estimasi Ukuran RAM: {param_count * 2 / 1024 / 1024:.1f} MB (bfloat16)")
    
    prompt = "Saya makan"
    print(f"\n3. Prompt: '{prompt}'")
    
    input_ids = tokenizer.encode(prompt).ids
    tokens = jnp.array([input_ids], dtype=jnp.int32)
    memory_state = variables.get('memory', {})
    params = variables['params']
    
    print("\n4. Mulai Generate Token (Inference Mode)...")
    start_time = time.time()
    
    # Gunakan fungsi yang sudah di-JIT untuk generasi cepat
    @jax.jit
    def generate_step(tokens_in, mem_in):
        (logits, _, _, _, _), mutated = model.apply(
            {'params': params, 'memory': mem_in},
            tokens_in,
            mutable=['memory']
        )
        next_tok = jnp.argmax(logits[0, -1])
        return next_tok, mutated.get('memory', mem_in)

    for i in range(20):
        t0 = time.time()
        next_tok, memory_state = generate_step(tokens, memory_state)
        # Block until computation finishes
        next_tok = int(next_tok) 
        tokens = jnp.concatenate([tokens, jnp.array([[next_tok]])], axis=1)
        
        if i == 0:
            print(f"   [Step 1 - Termasuk JIT Compile]: {time.time() - t0:.2f} detik (Lama karena pre-alloc)")
        elif i == 1:
            print(f"   [Step 2 - Seterusnya]: {time.time() - t0:.4f} detik per token (Sangat Cepat!)")
            
    total_time = time.time() - start_time
    output_text = tokenizer.decode(tokens[0].tolist())
    
    print(f"\n5. Hasil Generate (Acak karena bobot belum dilatih):")
    print(f"   {output_text}")
    print(f"\n✅ Inference selesai dengan mulus! (Total Waktu: {total_time:.2f}s)")

if __name__ == '__main__':
    main()
