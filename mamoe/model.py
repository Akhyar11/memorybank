import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

from .layers.normalization import RMSNorm
from .layers.embeddings import RoPE
from .layers.attention import CausalSelfAttention
from .layers.moe import MoELayer
from .memory.controller import MemoryController
from .memory.bank import MemoryBank

class MAMoEBlock(nn.Module):
    config: any
    
    @nn.compact
    def __call__(self, hidden_states, freqs_cis, attention_mask: Optional[jax.Array] = None):
        # Pre-norm Attention
        norm1 = RMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, name='input_layernorm')
        attn = CausalSelfAttention(config=self.config, name='self_attn')
        
        residual = hidden_states
        hidden_states = norm1(hidden_states)
        hidden_states = attn(hidden_states, freqs_cis, attention_mask)
        hidden_states = residual + hidden_states
        
        # Pre-norm MoE FFN
        norm2 = RMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, name='post_attention_layernorm')
        moe = MoELayer(config=self.config, name='moe')
        
        residual = hidden_states
        hidden_states = norm2(hidden_states)
        hidden_states, aux_loss = moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, aux_loss

class MAMoEForCausalLM(nn.Module):
    config: any
    
    def setup(self):
        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size, 
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )
        self.rope = RoPE(dim=self.config.head_dim, max_position_embeddings=self.config.max_position_embeddings)
        self.norm = RMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, name='norm')
        
        # We can't easily put blocks in a list via setup for nn.compact call without a loop inside __call__ anyway,
        # but flax supports variable collections or we can just define them inside __call__ with nn.compact
        
        self.memory_controller = MemoryController(config=self.config, name='memory_controller')
        self.memory_bank = MemoryBank(config=self.config, name='memory_bank')
        
    @nn.compact
    def __call__(
        self, 
        input_ids: jax.Array, 
        attention_mask: Optional[jax.Array] = None,
        is_eos: Optional[jax.Array] = None
    ):
        """
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, 1, 1, seq_len)
        is_eos: (batch_size,) boolean array indicating if the current step is the end of utterance.
                If provided, memory WRITE logic may be triggered.
        """
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        
        # Create positions
        positions = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch_size, seq_len))
        freqs_cis = self.rope(positions)
        
        # Pass through layers
        total_aux_loss = 0.0
        for i in range(self.config.num_hidden_layers):
            block = MAMoEBlock(config=self.config, name=f'layers_{i}')
            hidden_states, block_aux_loss = block(hidden_states, freqs_cis, attention_mask)
            total_aux_loss += block_aux_loss
            
        hidden_states = self.norm(hidden_states)
        
        # -- Memory Phase (UGMB) --
        h_eos = hidden_states[:, -1, :] # (batch_size, hidden_size)
        
        # 1. Gate evaluation (Independent READ and WRITE probabilities)
        read_prob, write_prob = self.memory_controller(h_eos)
        
        # 2. Memory interaction
        # fused_h_eos will be modified if READ is selected
        fused_h_eos = self.memory_bank(h_eos, read_prob, write_prob)
        
        # If the utterance just ended and write gate is active, we write to memory
        if is_eos is not None:
            # We conditionally write based on WRITE prob
            pass # In a real training loop, we use jax.lax.cond or a masked write
            
        # Replace the last token state with the fused state
        hidden_states = hidden_states.at[:, -1, :].set(fused_h_eos)
        
        # -- LM Head (Tied Weights) --
        # In tied weights, we use the embedding matrix transposed
        logits = jnp.matmul(hidden_states, self.embed_tokens.embedding.T)
        
        # Return probabilities for logging and aux_loss for balancing
        return logits, read_prob, write_prob, total_aux_loss
