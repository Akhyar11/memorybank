import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

from .layers.normalization import RMSNorm
from .layers.embeddings import RoPE
from .layers.attention import SelfAttention, CausalSelfAttention, CrossAttention
from .layers.moe import MoELayer
from .memory.controller import MemoryController
from .memory.bank import MemoryBank

class MAMoEEncoderBlock(nn.Module):
    config: any
    
    @nn.compact
    def __call__(self, hidden_states, freqs_cis, attention_mask: Optional[jax.Array] = None):
        norm1 = RMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, name='input_layernorm')
        attn = SelfAttention(config=self.config, name='self_attn')
        
        residual = hidden_states
        hidden_states = norm1(hidden_states)
        hidden_states = attn(hidden_states, freqs_cis, attention_mask)
        hidden_states = residual + hidden_states
        
        norm2 = RMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, name='post_attention_layernorm')
        moe = MoELayer(config=self.config, name='moe')
        
        residual = hidden_states
        hidden_states = norm2(hidden_states)
        hidden_states, aux_loss, f_i = moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, aux_loss, f_i

class MAMoEDecoderBlock(nn.Module):
    config: any
    
    @nn.compact
    def __call__(self, hidden_states, context_states, freqs_cis, attention_mask: Optional[jax.Array] = None, cross_attention_mask: Optional[jax.Array] = None):
        norm1 = RMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, name='input_layernorm')
        self_attn = CausalSelfAttention(config=self.config, name='self_attn')
        
        residual = hidden_states
        hidden_states = norm1(hidden_states)
        hidden_states = self_attn(hidden_states, freqs_cis, attention_mask)
        hidden_states = residual + hidden_states
        
        norm2 = RMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, name='cross_attn_layernorm')
        cross_attn = CrossAttention(config=self.config, name='cross_attn')
        
        residual = hidden_states
        hidden_states = norm2(hidden_states)
        hidden_states = cross_attn(hidden_states, context_states, cross_attention_mask)
        hidden_states = residual + hidden_states
        
        norm3 = RMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, name='post_attention_layernorm')
        moe = MoELayer(config=self.config, name='moe')
        
        residual = hidden_states
        hidden_states = norm3(hidden_states)
        hidden_states, aux_loss, f_i = moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, aux_loss, f_i

class MAMoEEncoder(nn.Module):
    config: any
    
    @nn.compact
    def __call__(self, embed_tokens, embed_proj, input_ids: jax.Array, attention_mask: Optional[jax.Array] = None):
        batch_size, seq_len = input_ids.shape
        raw_embeds = embed_tokens(input_ids)
        hidden_states = embed_proj(raw_embeds)
        
        rope = RoPE(dim=self.config.head_dim, max_position_embeddings=self.config.max_position_embeddings)
        positions = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch_size, seq_len))
        freqs_cis = rope(positions)
        
        total_aux_loss = 0.0
        total_f_i = jnp.zeros(self.config.num_experts)
        for i in range(self.config.num_hidden_layers):
            block = MAMoEEncoderBlock(config=self.config, name=f'layers_{i}')
            hidden_states, block_aux_loss, block_f_i = block(hidden_states, freqs_cis, attention_mask)
            total_aux_loss += block_aux_loss
            total_f_i += block_f_i
            
        norm = RMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, name='norm')
        hidden_states = norm(hidden_states)
        
        avg_f_i = total_f_i / self.config.num_hidden_layers
        return hidden_states, total_aux_loss, avg_f_i

class MAMoEDecoder(nn.Module):
    config: any
    
    @nn.compact
    def __call__(self, embed_tokens, embed_proj, decoder_input_ids: jax.Array, context_states: jax.Array, attention_mask: Optional[jax.Array] = None, cross_attention_mask: Optional[jax.Array] = None):
        batch_size, seq_len = decoder_input_ids.shape
        raw_embeds = embed_tokens(decoder_input_ids)
        hidden_states = embed_proj(raw_embeds)
        
        rope = RoPE(dim=self.config.head_dim, max_position_embeddings=self.config.max_position_embeddings)
        positions = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch_size, seq_len))
        freqs_cis = rope(positions)
        
        total_aux_loss = 0.0
        total_f_i = jnp.zeros(self.config.num_experts)
        for i in range(self.config.num_hidden_layers):
            block = MAMoEDecoderBlock(config=self.config, name=f'layers_{i}')
            hidden_states, block_aux_loss, block_f_i = block(hidden_states, context_states, freqs_cis, attention_mask, cross_attention_mask)
            total_aux_loss += block_aux_loss
            total_f_i += block_f_i
            
        norm = RMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps, name='norm')
        hidden_states = norm(hidden_states)
        
        avg_f_i = total_f_i / self.config.num_hidden_layers
        return hidden_states, total_aux_loss, avg_f_i

class MAMoEForConditionalGeneration(nn.Module):
    config: any
    
    def setup(self):
        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size, 
            features=self.config.embed_dim,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )
        self.embed_proj = nn.Dense(self.config.hidden_size, name='embed_proj')
        self.lm_head_proj = nn.Dense(self.config.embed_dim, name='lm_head_proj')
        
        self.encoder = MAMoEEncoder(config=self.config, name='encoder')
        self.decoder = MAMoEDecoder(config=self.config, name='decoder')
        
        self.memory_controller = MemoryController(config=self.config, name='memory_controller')
        self.memory_bank = MemoryBank(config=self.config, name='memory_bank')
        
    def __call__(
        self, 
        input_ids: jax.Array, 
        decoder_input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        decoder_attention_mask: Optional[jax.Array] = None,
        cross_attention_mask: Optional[jax.Array] = None,
        is_eos: Optional[jax.Array] = None
    ):
        encoder_hidden_states, enc_aux_loss, enc_f_i = self.encoder(
            self.embed_tokens, self.embed_proj, input_ids, attention_mask
        )
        
        h_prompt_eos = encoder_hidden_states[:, -1, :] 
        read_prob, write_prob = self.memory_controller(h_prompt_eos)
        fused_memory_context = self.memory_bank(h_prompt_eos, read_prob, write_prob) 
        
        fused_memory_context = jnp.expand_dims(fused_memory_context, axis=1)
        full_context_states = jnp.concatenate([fused_memory_context, encoder_hidden_states], axis=1)
        
        if cross_attention_mask is not None:
            mem_mask = jnp.zeros((cross_attention_mask.shape[0], cross_attention_mask.shape[1], cross_attention_mask.shape[2], 1))
            full_cross_attention_mask = jnp.concatenate([mem_mask, cross_attention_mask], axis=-1)
        else:
            full_cross_attention_mask = None

        decoder_hidden_states, dec_aux_loss, dec_f_i = self.decoder(
            self.embed_tokens, self.embed_proj, decoder_input_ids, full_context_states, decoder_attention_mask, full_cross_attention_mask
        )
        
        if is_eos is not None:
            h_decoder_eos = decoder_hidden_states[:, -1, :]
            self.memory_bank.write(h_decoder_eos, is_eos, write_prob)
            
        projected_states = self.lm_head_proj(decoder_hidden_states)
        logits = jnp.matmul(projected_states, self.embed_tokens.embedding.T)
        
        total_aux_loss = enc_aux_loss + dec_aux_loss
        avg_f_i = (enc_f_i + dec_f_i) / 2.0
        
        return logits, read_prob, write_prob, total_aux_loss, avg_f_i
