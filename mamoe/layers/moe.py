import jax
import jax.numpy as jnp
import flax.linen as nn
from .activations import SwiGLU

class ExpertFFN(nn.Module):
    config: any
    
    @nn.compact
    def __call__(self, x):
        # SwiGLU requires input projected to 2 * intermediate_size
        gate_up_proj = nn.Dense(2 * self.config.intermediate_size, use_bias=False, name='gate_up_proj')
        down_proj = nn.Dense(self.config.hidden_size, use_bias=False, name='down_proj')
        
        x = gate_up_proj(x)
        x = SwiGLU()(x)
        x = down_proj(x)
        return x

class MoERouter(nn.Module):
    config: any
    
    @nn.compact
    def __call__(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        gate_proj = nn.Dense(self.config.num_experts, use_bias=False, name='gate_proj')
        logits = gate_proj(x) # (batch_size, seq_len, num_experts)
        
        # Softmax over experts
        routing_weights = jax.nn.softmax(logits, axis=-1)
        
        # --- Load Balancing Metrics ---
        # f_i: fraction of tokens routed to each expert
        # P_i: mean routing probability for each expert
        # x is (batch, seq, dim). We flatten to (batch*seq, dim) for metric.
        routing_weights_flat = routing_weights.reshape(-1, self.config.num_experts)
        
        # For Top-K routing, the boolean assignment determines f_i
        top_k = self.config.num_experts_per_tok
        topk_weights, topk_indices = jax.lax.top_k(routing_weights, top_k)
        
        # One-hot assignment: (batch*seq_len, num_experts)
        # 1.0 if token assigned to expert, else 0.0
        mask1hot = jax.nn.one_hot(topk_indices, self.config.num_experts)
        # Sum across top-k choices: (batch*seq_len, num_experts)
        expert_mask = jnp.sum(mask1hot, axis=1)
        
        f_i = jnp.mean(expert_mask, axis=0) # (num_experts,)
        P_i = jnp.mean(routing_weights_flat, axis=0) # (num_experts,)
        
        aux_loss = self.config.router_aux_loss_coef * self.config.num_experts * jnp.sum(f_i * P_i)
        
        # If Top-1, no need to normalize across top-k dimension since k=1.
        if top_k > 1:
            topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
            
        return topk_weights, topk_indices, aux_loss

class MoELayer(nn.Module):
    config: any
    
    @nn.compact
    def __call__(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten for routing
        x = hidden_states.reshape(-1, hidden_size)
        
        router = MoERouter(config=self.config)
        routing_weights, selected_experts, aux_loss = router(x) # (batch*seq_len, top_k)
        
        # Output buffer
        final_hidden_states = jnp.zeros_like(x)
        
        # Iterate over all experts
        for expert_idx in range(self.config.num_experts):
            expert = ExpertFFN(config=self.config, name=f'expert_{expert_idx}')
            
            expert_mask = (selected_experts == expert_idx)
            expert_out = expert(x)
            
            for k in range(self.config.num_experts_per_tok):
                mask_k = expert_mask[:, k:k+1]
                weight_k = routing_weights[:, k:k+1]
                
                final_hidden_states += jnp.where(mask_k, expert_out * weight_k, 0.0)
                
        return final_hidden_states.reshape(batch_size, seq_len, hidden_size), aux_loss
