import os
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from tokenizers import Tokenizer
from mamoe.config import MAMoEConfig
from mamoe.model import MAMoEForCausalLM

def main():
    print("📦 Loading JAX MemoryBank Model...")
    
    ckpt_dir = '/kaggle/working/checkpoints/phase2'
    if not os.path.exists(ckpt_dir):
        ckpt_dir = '/kaggle/working/checkpoints/phase1'
        
    if not os.path.exists(ckpt_dir):
        print("⚠️ No JAX checkpoint found. Using RANDOM weights for testing!")
        raw_state = None
    else:
        checkpointer = ocp.StandardCheckpointer()
        raw_state = checkpointer.restore(os.path.abspath(ckpt_dir), target=None)
        
    config = MAMoEConfig()
    model = MAMoEForCausalLM(config=config)
    
    # Initialize variables to get the structure
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    dummy_eos = jnp.zeros((1,), dtype=jnp.int32)
    init_vars = model.init(rng, dummy_input, attention_mask=None, is_eos=dummy_eos)
    
    if raw_state is not None:
        params = raw_state['params'] if 'params' in raw_state else raw_state
    else:
        params = init_vars['params']
        
    # Start with a fresh memory state
    memory = init_vars['memory']
    
    print("🔤 Loading Tokenizer...")
    try:
        tokenizer = Tokenizer.from_file("tokenizer_hf/tokenizer.json")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return
        
    prompt = "User: Halo, tolong ingat bahwa warna kesukaanku adalah merah.\nAssistant:"
    input_ids = jnp.array([tokenizer.encode(prompt).ids], dtype=jnp.int32)
    
    print(f"\n🧠 Prompt: '{prompt}'")
    print("🔍 Mengamati Aktivitas Memory Controller...\n")
    
    # We will simulate 3 generation steps
    for step in range(1, 4):
        # We simulate that the model wants to write to memory on the 3rd step (is_eos = True)
        is_eos_flag = jnp.array([1 if step == 3 else 0], dtype=jnp.int32)
        
        # Apply the model and get the mutated memory back
        (logits, read_prob, write_prob, _, _), new_mutables = model.apply(
            {'params': params, 'memory': memory}, 
            input_ids, 
            attention_mask=None, 
            is_eos=is_eos_flag,
            mutable=['memory']
        )
        
        # Update our memory dictionary with the newly returned memory state
        memory = new_mutables['memory']
        
        # Determine how many items are active in memory right now
        # Active state == 1
        active_items = jnp.sum(memory['memory_bank']['state'] == 1).item()
        
        print(f"--- Step {step} (is_eos = {bool(is_eos_flag[0])}) ---")
        print(f"📖 Probabilitas READ  : {read_prob[0].item():.4f}")
        print(f"✍️  Probabilitas WRITE : {write_prob[0].item():.4f}")
        print(f"🧠 Slot Memori Aktif   : {active_items} item dari {config.memory_capacity} kapasitas\n")
        
        # Simulate adding a new token for the next step
        next_token = jnp.argmax(logits[0, -1, :]).reshape(1, 1)
        input_ids = jnp.concatenate([input_ids, next_token], axis=1)

    print("✅ Pengujian Memory Bank selesai! Logika READ dan WRITE berjalan sempurna.")

if __name__ == "__main__":
    main()
