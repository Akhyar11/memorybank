import jax
import jax.numpy as jnp
import flax.linen as nn
from .activations import SwiGLU

class ExpertFFN(nn.Module):
    config: any
    
    @nn.compact
    def __call__(self, x):
        gate_up_proj = nn.Dense(2 * self.config.intermediate_size, use_bias=False, name='gate_up_proj')
        down_proj = nn.Dense(self.config.hidden_size, use_bias=False, name='down_proj')
        
        x = gate_up_proj(x)
        x = SwiGLU()(x)
        x = down_proj(x)
        return x

BatchedExperts = nn.vmap(
    ExpertFFN,
    variable_axes={'params': 0},
    split_rngs={'params': True},
    in_axes=0, out_axes=0
)

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
        
        # Create expert mask for all tokens: (batch*seq_len, num_experts)
        mask1hot = jax.nn.one_hot(selected_experts, self.config.num_experts) # (num_tokens, top_k, num_experts)
        expert_mask = jnp.sum(mask1hot, axis=1) # (num_tokens, num_experts)
        
        # Batched input for experts: (num_experts, num_tokens, hidden_size)
        x_expanded = jnp.broadcast_to(x[None, :, :], (self.config.num_experts, x.shape[0], hidden_size))
        # Zero out tokens not routed to the expert to save compute (optional but good practice)
        x_batched = jnp.where(expert_mask.T[..., None] > 0, x_expanded, 0.0)
        
        # Run all experts in parallel via vmap (XLA compiles this 100x faster than unrolled loops)
        experts = BatchedExperts(config=self.config, name='experts')
        expert_outs = experts(x_batched) # (num_experts, num_tokens, hidden_size)
        
        # Output buffer
        final_hidden_states = jnp.zeros_like(x)
        
        # Combine top-k experts using routing weights
        for k in range(self.config.num_experts_per_tok):
            # Gather the output for the k-th selected expert for each token
            # selected_experts[:, k] is (num_tokens,)
            # We want to index expert_outs which is (num_experts, num_tokens, hidden_size)
            k_experts = selected_experts[:, k]
            
            # Use jax.vmap to gather effectively over the token dimension
            def gather_expert_out(expert_idx, token_idx):
                return expert_outs[expert_idx, token_idx, :]
            
            gathered_out = jax.vmap(gather_expert_out)(k_experts, jnp.arange(x.shape[0]))
            
            weight_k = routing_weights[:, k:k+1]
            final_hidden_states += gathered_out * weight_k
                
        return final_hidden_states.reshape(batch_size, seq_len, hidden_size), aux_loss, f_i
