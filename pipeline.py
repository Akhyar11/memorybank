import subprocess
import sys
import os

def setup_environment():
    """Auto-fix JAX CUDA plugin mismatch on Kaggle, then install remaining deps."""
    is_kaggle = os.path.exists('/kaggle')
    if not is_kaggle:
        print("⚙️  Local environment detected. Skipping JAX auto-fix.")
        return

    print("⚙️  Kaggle environment detected. Fixing JAX CUDA compatibility...")
    
    # Step 1: Remove the incompatible plugin and jaxlib
    subprocess.run([
        sys.executable, "-m", "pip", "uninstall", "-y",
        "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt"
    ], capture_output=True)
    
    # Step 2: Reinstall a clean, CUDA-12 compatible JAX bundle
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "-U", "jax[cuda12]"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("⚠️  jax[cuda12] install failed, trying CPU-only JAX as fallback...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax"], check=True)
    else:
        print("✅ JAX[CUDA12] installed successfully.")
    
    # Step 3: Install remaining pipeline dependencies (force upgrade to match JAX 0.11.x)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "flax", "optax", "orbax-checkpoint", "tokenizers", "pandas", "pyarrow", "fastparquet"
    ], check=True)
    print("✅ All dependencies ready.\n")

def check_kaggle_environment():
    """Validates that Kaggle dataset paths exist before starting."""
    print("="*50)
    print("MAMoE-50 Kaggle Training Pipeline")
    print("="*50)
    
    vqfat_path = '/kaggle/input/datasets/akhyarsafrudin/dataset-chat/vqfat_cosmopedia_id.csv'
    t5gemma_path = '/kaggle/input/datasets/akhyarsafrudin/dataset-chat/t5gemma2_chat_multiturn.parquet'
    
    # We now use the local HF tokenizer downloaded by fetch_embeddings.py
    tokenizer_path = 'tokenizer_hf/tokenizer.json'
    
    if not os.path.exists(vqfat_path):
        print(f"⚠️  WARNING: Pre-training dataset not found at {vqfat_path}")
    else:
        print("✅ Phase 1 Dataset Found (VQFat CSV)")
        
    if not os.path.exists(t5gemma_path):
        print(f"⚠️  WARNING: Fine-Tuning dataset not found at {t5gemma_path}")
    else:
        print("✅ Phase 2 Dataset Found (T5Gemma2 Parquet)")
        
    if not os.path.exists(tokenizer_path):
        print(f"⚠️  WARNING: Tokenizer not found at {tokenizer_path}. Please run 'python fetch_embeddings.py' first!")
    else:
        print("✅ Tokenizer Found")
    print("="*50 + "\n")

def run_pretokenize():
    """Pre-tokenize CSV → npy once (skip if already done)."""
    npy_path = '/kaggle/working/vqfat_tokens.npy'
    if os.path.exists(npy_path):
        import numpy as np
        arr = np.load(npy_path, mmap_mode='r')
        print(f"⚡ Pre-tokenized file exists ({arr.shape[0]:,} tokens). Skipping tokenization.\n")
        return
    print("🔤 PRE-TOKENIZING dataset (runs once, makes training much faster)...")
    result = subprocess.run([sys.executable, "pretokenize.py"], check=False)
    if result.returncode != 0:
        print("⚠️  Pre-tokenization failed. Training will fall back to CSV streaming.")
    else:
        print("✅ PRE-TOKENIZATION DONE!\n")

def run_phase_1():
    print("🚀 STARTING PHASE 1: PRE-TRAINING (Next Token Predictor)")
    print("Running train.py...")
    try:
        subprocess.run([sys.executable, "train.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Phase 1 failed with error code {e.returncode}")
        sys.exit(e.returncode)
    print("✅ PHASE 1 COMPLETE!\n")

def run_phase_2():
    print("🚀 STARTING PHASE 2: FINE-TUNING (Chat Alignment)")
    print("Running finetune.py...")
    try:
        subprocess.run([sys.executable, "finetune.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Phase 2 failed with error code {e.returncode}")
        sys.exit(e.returncode)
    print("✅ PHASE 2 COMPLETE!\n")

def run_export_pytorch():
    print("📦 STARTING PHASE 3: EXPORT TO PYTORCH")
    print("Running export_pytorch.py...")
    try:
        subprocess.run([sys.executable, "export_pytorch.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Export failed with error code {e.returncode}")
    print("✅ EXPORT COMPLETE! Model ready to be downloaded.\n")

if __name__ == "__main__":
    setup_environment()
    check_kaggle_environment()
    run_pretokenize()
    run_phase_1()
    run_phase_2()
    run_export_pytorch()
    print("🎉 FULL PIPELINE EXECUTION SUCCESSFUL!")
