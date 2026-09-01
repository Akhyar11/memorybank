from dataclasses import dataclass, field
import json

@dataclass
class MAMoEConfig:
    vocab_size: int = 31923         # IndoBERT vocab size
    hidden_size: int = 128          # d = 128
    embed_dim: int = 768            # IndoBERT embed dim
    freeze_embeddings: bool = False # Allow embeddings to train
    num_hidden_layers: int = 8      # L = 8
    num_attention_heads: int = 4    # Heads = 4 (32 dim)
    head_dim: int = 32
    intermediate_size: int = 512    # d_ff = 512
    num_experts: int = 64           # E = 64
    num_experts_per_tok: int = 4    # Top-4 routing
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    # Memory Config
    memory_capacity: int = 10000   
    memory_top_k: int = 4          
    memory_dim: int = 128          # Matches hidden_size
    memory_threshold: float = 0.8  # Tau (relevance/update threshold)
    memory_read_threshold: float = 0.5
    memory_write_threshold: float = 0.5
    memory_update_threshold: float = 0.8
    
    # Memory Meta-data & Scoring Hyperparameters
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

    def __post_init__(self):
        # Validation checks as per requirements
        assert self.hidden_size == self.num_attention_heads * self.head_dim, \
            f"hidden_size ({self.hidden_size}) must equal num_attention_heads ({self.num_attention_heads}) * head_dim ({self.head_dim})"
        
        assert self.memory_dim == self.hidden_size, \
            f"memory_dim ({self.memory_dim}) must equal hidden_size ({self.hidden_size})"
            
        assert self.num_experts_per_tok <= self.num_experts, \
            f"num_experts_per_tok ({self.num_experts_per_tok}) must be <= num_experts ({self.num_experts})"
            
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)
            
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
