# MAMoE-50: Memory-Augmented Mixture-of-Experts

**MAMoE-50** is a highly efficient, from-scratch ~50M parameter causal language model built in pure **JAX/Flax**. It features a specialized **Neural Persistent Memory Layer** allowing the network to explicitly read, write, decay, and update episodic memories across conversations.

Designed specifically for constrained environments (e.g. Kaggle Notebooks, Edge SLMs), it combines **Top-1 Routing Mixture-of-Experts (MoE)** and an utterance-level **Memory Bank** to drastically reduce active compute parameters while retaining a massive capacity for knowledge.

## Features
- **50M Total Parameters | 5M Active Parameters per Token**: Achieved via 16-expert Top-1 MoE architecture.
- **Neural Persistent Memory Bank**: A stateless/functional memory array that stores representations with explicit metadata (`Importance`, `Confidence`, `Recency`, `State`).
- **Load Balanced MoE**: Built-in auxiliary loss to prevent *expert collapse*.
- **100% Pure JAX/Flax**: Highly optimized for XLA compilation on TPUs and GPUs.

## Architecture

```text
Input Tokens -> Embeddings (32K Vocab)
    │
    ▼
[ Transformer Block x 8 ]
    ├── RMSNorm
    ├── Causal Self-Attention (6 heads, dim=64)
    ├── RMSNorm
    └── MoE FFN (16 Experts, Top-1 Routing, Load Balanced)
    │
    ▼
(h_EOS - Utterance End State)
    │
    ▼
[ Neural Memory Bank (UGMB) ]
    ├── Dual Sigmoid Gates (Independent READ & WRITE)
    ├── Scoring (CosSim + Importance + Recency + Confidence)
    ├── Write Strategy (Interpolation Update vs Dormant Overwrite)
    └── State Decay (Active -> Dormant -> Expired)
```

## Setup & Training

### 1. Installation
Install the required dependencies (assuming CUDA is pre-installed):
```bash
pip install -r requirements.txt
```

### 2. Pre-Training (Optax)
The training loop is written in pure Optax with XLA JIT compilation.
To run the training loop locally:
```bash
python train.py
```

### 3. Kaggle Deployment
This repository is pre-configured for Kaggle. See `kaggle_train.ipynb` for a ready-to-run notebook that fetches the dataset (e.g., VQFat Cosmopedia) from `/kaggle/input` and executes the JAX training loop on Kaggle's P100 or T4x2 instances.

## License
MIT License
