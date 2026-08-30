import subprocess
import sys
import os

def check_kaggle_environment():
    """Validates that Kaggle dataset paths exist before starting."""
    print("="*50)
    print("MAMoE-50 Kaggle Training Pipeline")
    print("="*50)
    
    vqfat_path = '/kaggle/input/vqfat-indonesian-corpus/vqfat_cosmopedia_id.csv'
    t5gemma_path = '/kaggle/input/t5gemma2-indonesia-chat/t5gemma2_chat_multiturn.parquet'
    
    if not os.path.exists(vqfat_path):
        print(f"⚠️  WARNING: Pre-training dataset not found at {vqfat_path}")
        print("Ensure you have added the 'vqfat-indonesian-corpus' dataset to your Kaggle Notebook.")
    else:
        print("✅ Phase 1 Dataset Found (VQFat CSV)")
        
    if not os.path.exists(t5gemma_path):
        print(f"⚠️  WARNING: Fine-Tuning dataset not found at {t5gemma_path}")
        print("Ensure you have added the 't5gemma2-indonesia-chat' dataset to your Kaggle Notebook.")
    else:
        print("✅ Phase 2 Dataset Found (T5Gemma2 Parquet)")
    print("="*50 + "\n")

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

if __name__ == "__main__":
    check_kaggle_environment()
    run_phase_1()
    run_phase_2()
    print("🎉 FULL PIPELINE EXECUTION SUCCESSFUL!")
