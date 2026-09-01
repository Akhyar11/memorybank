import jax
import jax.numpy as jnp
import flax.linen as nn
from .activations import SwiGLU

class ExpertFFN(nn.Module):
    # This class is kept for compatibility but we will implement
    # the sparse dispatch directly in MoELayer.
    config: any

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
        
        # One-hot assignment: (batch*seq_len, top_k, num_experts)
        # 1.0 if token assigned to expert, else 0.0
        mask1hot = jax.nn.one_hot(topk_indices, self.config.num_experts)
        # Sum across top-k choices: (batch*seq_len, num_experts)
        expert_mask = jnp.sum(mask1hot, axis=1)
        
        # f_i is the fraction of tokens routed to each expert.
        # Since each token routes to top_k experts, the total assignments is (batch*seq_len) * top_k
        f_i = jnp.sum(expert_mask, axis=0) / (expert_mask.shape[0] * top_k) # (num_experts,)
        P_i = jnp.mean(routing_weights_flat, axis=0) # (num_experts,)
        
        aux_loss = self.config.router_aux_loss_coef * self.config.num_experts * jnp.sum(f_i * P_i)
        
        # If Top-1, no need to normalize across top-k dimension since k=1.
        if top_k > 1:
            topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
            
        return topk_weights, topk_indices, aux_loss, f_i  # f_i: (num_experts,) fraction per expert

class MoELayer(nn.Module):
    config: any
    
    @nn.compact
    def __call__(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten for routing
        x = hidden_states.reshape(-1, hidden_size)
        
        router = MoERouter(config=self.config)
        routing_weights, selected_experts, aux_loss, f_i = router(x)  # (batch*seq_len, top_k)
        
        # Initialize stacked expert weights
        gate_up_weights = self.param('gate_up_proj_weight', 
                                     nn.initializers.lecun_normal(), 
                                     (self.config.num_experts, hidden_size, 2 * self.config.intermediate_size))
        down_weights = self.param('down_proj_weight', 
                                  nn.initializers.lecun_normal(), 
                                  (self.config.num_experts, self.config.intermediate_size, hidden_size))
        
        # selected_experts is (batch*seq_len, top_k)
        # Gather weights for the selected experts
        # Shape: (batch*seq_len, top_k, hidden_size, 2*intermediate_size)
        w_gate_up = gate_up_weights[selected_experts] 
        # Shape: (batch*seq_len, top_k, intermediate_size, hidden_size)
        w_down = down_weights[selected_experts]
        
        # Expand x to compute against top_k experts: (batch*seq_len, 1, hidden_size)
        x_expanded = x[:, None, :]
        
        # Compute Gate & Up Proj
        # (batch*seq_len, 1, hidden_size) @ (batch*seq_len, top_k, hidden_size, 2*intermediate_size) 
        # -> (batch*seq_len, top_k, 2*intermediate_size)
        h = jnp.einsum('b k d, b k d h -> b k h', x_expanded.repeat(self.config.num_experts_per_tok, axis=1), w_gate_up)
        
        # SwiGLU
        h1, h2 = jnp.split(h, 2, axis=-1)
        h_act = SwiGLU()(h1) * h2
        
        # Compute Down Proj
        # (batch*seq_len, top_k, intermediate_size) @ (batch*seq_len, top_k, intermediate_size, hidden_size)
        # -> (batch*seq_len, top_k, hidden_size)
        expert_outs = jnp.einsum('b k i, b k i d -> b k d', h_act, w_down)
        
        # Weight by routing probs
        # routing_weights is (batch*seq_len, top_k)
        final_hidden_states = jnp.sum(expert_outs * routing_weights[..., None], axis=1)
        
        return final_hidden_states.reshape(batch_size, seq_len, hidden_size), aux_loss, f_i
