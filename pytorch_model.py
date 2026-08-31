import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pytorch_memory import MemoryBank

class MAMoEConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 31923)
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.embed_dim = kwargs.get('embed_dim', 768)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 8)
        self.num_attention_heads = kwargs.get('num_attention_heads', 4)
        self.head_dim = kwargs.get('head_dim', 64)
        self.intermediate_size = kwargs.get('intermediate_size', 512)
        self.num_experts = kwargs.get('num_experts', 16)
        self.num_experts_per_tok = kwargs.get('num_experts_per_tok', 1)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 2048)
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-6)
        self.rope_theta = kwargs.get('rope_theta', 10000.0)

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
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)

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

    def forward(self, x, cos, sin):
        bsz, seq_len, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = (q * cos) + (torch.cat([-q[..., self.head_dim//2:], q[..., :self.head_dim//2]], dim=-1) * sin)
        k = (k * cos) + (torch.cat([-k[..., self.head_dim//2:], k[..., :self.head_dim//2]], dim=-1) * sin)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).view(1, 1, seq_len, seq_len)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        o = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(o)

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
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])

    def forward(self, x):
        # Simplistic MoE routing for PyTorch load testing
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, 1, dim=-1)
        
        output = torch.zeros_like(x)
        # Inefficient loop just for structural compatibility
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).float()
            if mask.sum() > 0:
                expert_out = expert(x)
                output += (expert_out * mask)
                
        return output

class MemoryController(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.read_gate = nn.Linear(config.hidden_size, 1, bias=True)
        self.write_gate = nn.Linear(config.hidden_size, 1, bias=True)
        
    def forward(self, x):
        return torch.sigmoid(self.read_gate(x)), torch.sigmoid(self.write_gate(x))

class MAMoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = MoELayer(config)

    def forward(self, x, cos, sin):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin)
        h = h + self.moe(self.post_attention_layernorm(h))
        return h

class MAMoEForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim)
        self.embed_proj = nn.Linear(config.embed_dim, config.hidden_size)
        self.lm_head_proj = nn.Linear(config.hidden_size, config.embed_dim)
        self.layers = nn.ModuleList([MAMoEBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rope = RoPE(config.head_dim, config.rope_theta)
        self.memory_controller = MemoryController(config)
        self.memory_bank = MemoryBank(config)

    def forward(self, input_ids, mem_state=None, is_eos=False):
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)
        x = self.embed_proj(x)
        cos, sin = self.rope(seq_len)
        cos = cos.view(1, 1, seq_len, -1)
        sin = sin.view(1, 1, seq_len, -1)
        
        for layer in self.layers:
            x = layer(x, cos, sin)
            
        x = self.norm(x)
        
        write_prob_val = None
        if mem_state is not None:
            read_prob, write_prob = self.memory_controller(x)
            memory_output, mem_state = self.memory_bank.read(x, mem_state)
            x = x + read_prob * memory_output
            
            if is_eos:
                # Use the last token for write operation
                h_eos = x[:, -1, :]
                w_prob = write_prob[:, -1, :]
                mem_state = self.memory_bank.write(h_eos, w_prob, mem_state)
                write_prob_val = w_prob
                
        x_proj = self.lm_head_proj(x)
        logits = torch.matmul(x_proj, self.embed_tokens.weight.T)
        
        if mem_state is not None:
            return logits, mem_state, write_prob_val
        return logits
