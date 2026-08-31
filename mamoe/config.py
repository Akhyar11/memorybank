from dataclasses import dataclass

@dataclass
class MAMoEConfig:
    vocab_size: int = 31923         # IndoBERT vocab size
    hidden_size: int = 256          # d = 256
    embed_dim: int = 768            # IndoBERT embed dim
    freeze_embeddings: bool = True  # Freeze embeddings during training
    num_hidden_layers: int = 8      # L = 8
    num_attention_heads: int = 4    # Heads = 4 (64 dim)
    head_dim: int = 64
    intermediate_size: int = 512    # d_ff = 512
    num_experts: int = 64           # E = 64
    num_experts_per_tok: int = 4    # Top-4 routing
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    # Memory Config
    memory_capacity: int = 10000   
    memory_top_k: int = 4          
    memory_dim: int = 256          # Matches hidden_size
    memory_threshold: float = 0.8  # Tau (relevance/update threshold)
    
    # Memory Meta-data & Scoring Hyperparameters
    # We define default initial values for alpha, beta, gamma, delta
    # Score = alpha*Sim + beta*Importance + gamma*Recency + delta*Confidence
    mem_alpha: float = 1.0
    mem_beta: float = 0.5
    mem_gamma: float = 0.1
    mem_delta: float = 0.2
    
    # Decay & Reinforcement
    mem_decay_rate: float = 0.001       # lambda for recency R = e^(-lambda * dt)
    mem_importance_protection: float = 0.5 # rho for EffectiveDecay = R * (1 + rho * I)
    mem_reinforcement_rate: float = 0.1 # eta_a for access boost
    
    # MoE Load Balancing
    router_aux_loss_coef: float = 0.01

    # Dropout (optional for training)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    # Data type
    dtype: str = "float32"
