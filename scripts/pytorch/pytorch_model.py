
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from scripts.pytorch.pytorch_memory import MemoryBank
from scripts.pytorch.pytorch_utils import extract_last_valid_hidden, make_key_padding_mask, make_causal_mask

from mamoe.configuration_mamoe import MAMoEConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight

class RoPE(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # For pairwise RoPE (Llama-style), we repeat interleaving rather than cat
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        return torch.cos(emb), torch.sin(emb)
        
def rotate_half(x):
    """Rotates half the hidden dims of the input pairwise."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, attention_mask, cos, sin):
        bsz, seq_len, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE pairwise
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        o = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(o)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, attention_mask, cos, sin, past_key_value=None):
        bsz, seq_len, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE pairwise
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        present_key_value = (k, v)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        o = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(o), present_key_value

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, context, attention_mask, past_key_value=None):
        bsz, q_len, _ = x.shape
        q = self.q_proj(x)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if past_key_value is not None:
            k, v = past_key_value
        else:
            _, kv_len, _ = context.shape
            k = self.k_proj(context)
            v = self.v_proj(context)
            k = k.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
            
        present_key_value = (k, v)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        o = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(o), present_key_value

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        x = self.gate_up_proj(x)
        x1, x2 = x.chunk(2, dim=-1)
        return self.down_proj(F.silu(x1) * x2)

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])

    def forward(self, x):
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        top_k = self.config.num_experts_per_tok
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        
        if top_k > 1:
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
            
        bsz, seq_len, hidden = x.shape
        if bsz == 1 and seq_len == 1:
            if hasattr(self, 'use_stacked') and self.use_stacked:
                # 100% Vectorized Fast Path without CPU sync
                selected_gate_up = self.stacked_gate_up[top_k_indices[0, 0]]
                selected_down = self.stacked_down[top_k_indices[0, 0]]
                
                h = torch.einsum('b s h, k i h -> k s i', x, selected_gate_up)
                h1, h2 = h.chunk(2, dim=-1)
                h_act = F.silu(h1) * h2
                
                out = torch.einsum('k s i, k h i -> k s h', h_act, selected_down)
                probs = top_k_probs[0, 0].view(top_k, 1, 1)
                output = (out * probs).sum(dim=0, keepdim=True)
                return output
            else:
                # Fast path for single token generation (if KV cache is used but not stacked)
                output = torch.zeros_like(x)
                for k in range(top_k):
                    idx = top_k_indices[0, 0, k].item()
                    prob = top_k_probs[0, 0, k]
                    output += self.experts[idx](x) * prob
                return output
            
        x_flat = x.view(-1, x.shape[-1])
        top_k_indices_flat = top_k_indices.view(-1, top_k)
        top_k_probs_flat = top_k_probs.view(-1, top_k)
        output_flat = torch.zeros_like(x_flat)
        
        for k in range(top_k):
            indices_k = top_k_indices_flat[:, k]
            probs_k = top_k_probs_flat[:, k].unsqueeze(-1)
            
            unique_experts = indices_k.unique().tolist()
            for i in unique_experts:
                mask_bool = (indices_k == i)
                tokens_for_expert = x_flat[mask_bool]
                
                if hasattr(self, 'use_stacked') and self.use_stacked:
                    gate_weight = self.stacked_gate_up[i]
                    down_weight = self.stacked_down[i]
                    h = F.linear(tokens_for_expert, gate_weight)
                    h1, h2 = h.chunk(2, dim=-1)
                    h_act = F.silu(h1) * h2
                    expert_out = F.linear(h_act, down_weight)
                else:
                    expert_out = self.experts[i](tokens_for_expert)
                    
                output_flat[mask_bool] += expert_out * probs_k[mask_bool]
                
        return output_flat.view(*x.shape)

class MemoryController(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.read_gate = nn.Linear(config.hidden_size, 1, bias=True)
        self.write_gate = nn.Linear(config.hidden_size, 1, bias=True)
        
    def forward(self, x):
        return torch.sigmoid(self.read_gate(x)), torch.sigmoid(self.write_gate(x))

class MAMoEEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = MoELayer(config)

    def forward(self, x, attention_mask, cos, sin):
        h = x + self.self_attn(self.input_layernorm(x), attention_mask, cos, sin)
        h = h + self.moe(self.post_attention_layernorm(h))
        return h

class MAMoEDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.cross_attn_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.cross_attn = CrossAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = MoELayer(config)

    def forward(self, x, context, attention_mask, cross_attention_mask, cos, sin, past_key_value=None):
        self_kv = past_key_value[0] if past_key_value is not None else None
        cross_kv = past_key_value[1] if past_key_value is not None else None
        
        attn_out, self_kv = self.self_attn(self.input_layernorm(x), attention_mask, cos, sin, self_kv)
        h = x + attn_out
        
        cross_out, cross_kv = self.cross_attn(self.cross_attn_layernorm(h), context, cross_attention_mask, cross_kv)
        h = h + cross_out
        
        h = h + self.moe(self.post_attention_layernorm(h))
        return h, (self_kv, cross_kv)

class MAMoEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MAMoEEncoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
    def forward(self, x, attention_mask, cos, sin):
        for layer in self.layers:
            x = layer(x, attention_mask, cos, sin)
        return self.norm(x)
        
class MAMoEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MAMoEDecoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
    def forward(self, x, context, attention_mask, cross_attention_mask, cos, sin, past_key_values=None, use_cache=False):
        next_decoder_cache = () if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, layer_past = layer(x, context, attention_mask, cross_attention_mask, cos, sin, layer_past)
            if use_cache:
                next_decoder_cache += (layer_past,)
        return self.norm(x), next_decoder_cache

class MAMoEForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, getattr(config, 'embed_dim', config.hidden_size))
        self.embed_proj = nn.Linear(getattr(config, 'embed_dim', config.hidden_size), config.hidden_size, bias=False)
        self.encoder = MAMoEEncoder(config)
        self.decoder = MAMoEDecoder(config)
        self.rope = RoPE(config.head_dim)
        self.lm_head_proj = nn.Linear(config.hidden_size, getattr(config, 'embed_dim', config.hidden_size), bias=False)
        self.memory_controller = MemoryController(config)
        self.memory_bank = MemoryBank(config)

    def stack_experts(self):
        """Stacks expert weights to avoid CPU-GPU syncs during inference."""
        for name, module in self.named_modules():
            if isinstance(module, MoELayer):
                gate_up_weight = torch.stack([e.gate_up_proj.weight.data for e in module.experts])
                down_weight = torch.stack([e.down_proj.weight.data for e in module.experts])
                module.stacked_gate_up = nn.Parameter(gate_up_weight, requires_grad=False)
                module.stacked_down = nn.Parameter(down_weight, requires_grad=False)
                module.use_stacked = True
                
                # Delete old weights to prevent OOM on small GPUs
                for e in module.experts:
                    del e.gate_up_proj.weight
                    del e.down_proj.weight

    def forward(self, input_ids, decoder_input_ids, attention_mask=None, decoder_attention_mask=None, mem_state=None, is_eos=False, encoder_outputs=None, past_key_values=None, use_cache=False):
        if encoder_outputs is None:
            # 1. Encoder Pass
            bsz, enc_seq_len = input_ids.shape
            x_enc = self.embed_proj(self.embed_tokens(input_ids))
            cos_enc, sin_enc = self.rope(enc_seq_len)
            cos_enc = cos_enc.view(1, 1, enc_seq_len, -1)
            sin_enc = sin_enc.view(1, 1, enc_seq_len, -1)
            
            enc_additive_mask = make_key_padding_mask(attention_mask, dtype=x_enc.dtype)
            
            encoder_hidden_states = self.encoder(x_enc, enc_additive_mask, cos_enc, sin_enc)
            
            # 2. Memory READ Phase
            h_prompt_eos = extract_last_valid_hidden(encoder_hidden_states, attention_mask=attention_mask)
            read_prob, write_prob = self.memory_controller(h_prompt_eos)
            
            write_prob_val = None
            if mem_state is not None:
                memory_output, mem_state = self.memory_bank.read(h_prompt_eos, mem_state)
                fused_memory_context = memory_output * read_prob
                
                fused_memory_context = fused_memory_context.unsqueeze(1) # (bsz, 1, hidden)
                full_context_states = torch.cat([fused_memory_context, encoder_hidden_states], dim=1)
            else:
                full_context_states = encoder_hidden_states
        else:
            full_context_states, write_prob, mem_state = encoder_outputs
            write_prob_val = None
            
        # 3. Decoder Pass
        bsz, dec_seq_len = decoder_input_ids.shape
        x_dec = self.embed_proj(self.embed_tokens(decoder_input_ids))
        
        past_len = past_key_values[0][0][0].shape[2] if past_key_values is not None else 0
        total_len = past_len + dec_seq_len
        
        cos_dec, sin_dec = self.rope(total_len)
        cos_dec = cos_dec.view(1, 1, total_len, -1)[:, :, past_len:total_len, :]
        sin_dec = sin_dec.view(1, 1, total_len, -1)[:, :, past_len:total_len, :]
        
        if attention_mask is not None:
            mem_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)
            cross_attention_base_mask = torch.cat([mem_mask, attention_mask], dim=-1)
            full_cross_attention_mask = make_key_padding_mask(cross_attention_base_mask, dtype=x_dec.dtype)
        else:
            full_cross_attention_mask = None
            
        causal_mask = make_causal_mask(dec_seq_len, past_len, device=x_dec.device, dtype=x_dec.dtype)
        dec_padding_mask = make_key_padding_mask(decoder_attention_mask, dtype=x_dec.dtype)
        
        if dec_padding_mask is not None:
            dec_additive_mask = causal_mask + dec_padding_mask
        else:
            dec_additive_mask = causal_mask
        
        decoder_hidden_states, next_decoder_cache = self.decoder(x_dec, full_context_states, dec_additive_mask, full_cross_attention_mask, cos_dec, sin_dec, past_key_values=past_key_values, use_cache=use_cache)
        
        # 4. Memory WRITE Phase
        if mem_state is not None and is_eos:
            h_decoder_eos = extract_last_valid_hidden(decoder_hidden_states, attention_mask=decoder_attention_mask)
            mem_state = self.memory_bank.write(h_decoder_eos, write_prob, mem_state)
            write_prob_val = write_prob
                
        # 5. LM Head
        x_proj = self.lm_head_proj(decoder_hidden_states)
        logits = torch.matmul(x_proj, self.embed_tokens.weight.T)
        
        if use_cache:
            if mem_state is not None:
                return logits, mem_state, write_prob_val, next_decoder_cache
            return logits, next_decoder_cache
            
        if mem_state is not None:
            return logits, mem_state, write_prob_val
        return logits
