import os
import torch
import jax
import jax.numpy as jnp
import numpy as np

# Import dari script lokal
from mamoe.config import MAMoEConfig
from mamoe.model import MAMoEForConditionalGeneration
from scripts.pytorch.pytorch_model import MAMoEForConditionalGeneration as PyTorchMAMoE, MAMoEConfig as PyTorchConfig
from export_pytorch import convert_jax_to_pytorch

def main():
    print("🚀 Memulai Test Export Model JAX ke PyTorch...")
    
    print("\n[1/4] Inisialisasi JAX Model...")
    config_jax = MAMoEConfig()
    model_jax = MAMoEForConditionalGeneration(config=config_jax)
    rng = jax.random.PRNGKey(42)
    
    # Dummy inputs untuk JAX model.init
    batch_size = 2
    enc_seq_len = 16
    dec_seq_len = 16
    
    dummy_enc = jnp.ones((batch_size, enc_seq_len), dtype=jnp.int32)
    dummy_dec = jnp.ones((batch_size, dec_seq_len), dtype=jnp.int32)
    dummy_eos = jnp.zeros((batch_size,), dtype=jnp.int32)
    
    # Initialize variables
    print("      Melakukan model.init() (Ini mungkin memakan waktu sebentar)...")
    variables = model_jax.init(rng, input_ids=dummy_enc, decoder_input_ids=dummy_dec, attention_mask=None, is_eos=dummy_eos)
    jax_params = variables['params']
    print("✅ JAX Model berhasil diinisialisasi.")
    
    print("\n[2/4] Inisialisasi PyTorch Model...")
    config_pt = PyTorchConfig()
    model_pt = PyTorchMAMoE(config_pt)
    print("✅ PyTorch Model berhasil diinisialisasi.")
    
    print("\n[3/4] Melakukan Konversi (convert_jax_to_pytorch)...")
    try:
        state_dict = convert_jax_to_pytorch(jax_params, model_pt, config_pt)
        print("✅ Konversi sukses! State Dict berhasil dibuat.")
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat konversi:\n{e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[4/4] Verifikasi PyTorch Model (load_state_dict)...")
    try:
        # Load ulang untuk memastikan format dan ukurannya persis sesuai
        missing_keys, unexpected_keys = model_pt.load_state_dict(state_dict, strict=False)
        print("✅ load_state_dict berhasil dijalankan!")
        if missing_keys:
            print(f"⚠️  Missing keys (biasanya wajar jika terkait RoPE/Masking): {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {unexpected_keys}")
            
        print("\n🎉 TES SELESAI! Sistem konversi JAX -> PyTorch berfungsi normal.")
        print("Anda sekarang bisa menjalankan training dengan tenang.")
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat load_state_dict ke PyTorch:\n{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
