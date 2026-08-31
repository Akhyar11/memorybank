
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_EXPIRED = 0
STATE_ACTIVE = 1
STATE_DORMANT = 2

class MemoryBank(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        capacity = config.memory_capacity
        dim = config.memory_dim
        
        # Encoders
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.i_proj = nn.Linear(dim, 1)
        
        self.fusion_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        
    def init_state(self, bsz=1, device='cpu'):
        capacity = self.config.memory_capacity
        dim = self.config.memory_dim
        return {
            'keys': torch.zeros(bsz, capacity, dim, device=device),
            'vals': torch.zeros(bsz, capacity, dim, device=device),
            'importance': torch.zeros(bsz, capacity, device=device),
            'confidence': torch.zeros(bsz, capacity, device=device),
            'last_access': torch.zeros(bsz, capacity, dtype=torch.long, device=device),
            'state': torch.zeros(bsz, capacity, dtype=torch.long, device=device),
            'global_step': torch.zeros(bsz, dtype=torch.long, device=device),
            'ptr': torch.zeros(bsz, dtype=torch.long, device=device)
        }

    def read(self, h_eos, mem_state):
        step = mem_state['global_step']
        keys = mem_state['keys']
        vals = mem_state['vals']
        importance = mem_state['importance']
        confidence = mem_state['confidence']
        last_access = mem_state['last_access']
        state = mem_state['state']
        
        bsz, dim = h_eos.shape
        
        q = self.q_proj(h_eos) # (bsz, dim)
        
        q_norm = F.normalize(q, p=2, dim=-1, eps=1e-8)
        k_norm = F.normalize(keys, p=2, dim=-1, eps=1e-8)
        
        # (bsz, 1, dim) @ (bsz, dim, capacity) -> (bsz, 1, capacity)
        sim = torch.bmm(q_norm.unsqueeze(1), k_norm.transpose(1, 2)).squeeze(1) # (bsz, capacity)
        
        dt = (step.unsqueeze(-1) - last_access).clamp(min=0).float()
        lam = getattr(self.config, 'mem_decay_rate', 0.001)
        recency = torch.exp(-lam * dt)
        
        alpha = getattr(self.config, 'mem_alpha', 1.0)
        beta = getattr(self.config, 'mem_beta', 0.5)
        gamma = getattr(self.config, 'mem_gamma', 0.1)
        delta = getattr(self.config, 'mem_delta', 0.1)
        
        # Broadcast metadata
        score = (alpha * sim + 
                 beta * importance + 
                 gamma * recency + 
                 delta * confidence)
                 
        mask = (state != STATE_EXPIRED)
        score = score.masked_fill(~mask, -1e9)
        
        k = self.config.memory_top_k
        topk_scores, topk_indices = torch.topk(score, k, dim=-1)
        
        tau = self.config.memory_threshold
        valid_mask = topk_scores > tau
        
        filtered_scores = topk_scores.masked_fill(~valid_mask, -1e9)
        attn_weights = F.softmax(filtered_scores, dim=-1)
        attn_weights = attn_weights * valid_mask.float()
        
        # Gather top-k values: (bsz, k, dim)
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, dim)
        topk_vals = torch.gather(vals, 1, expanded_indices)
        
        read_result = torch.sum(attn_weights.unsqueeze(-1) * topk_vals, dim=1)
        
        # Access Reinforcement
        for b in range(bsz):
            valid_idx = topk_indices[b][valid_mask[b]]
            if valid_idx.numel() > 0:
                valid_idx = torch.unique(valid_idx)
                mem_state['last_access'][b, valid_idx] = step[b]
                mem_state['importance'][b, valid_idx] += 0.05
                
        return read_result, mem_state

    def write(self, h_eos, write_prob, mem_state):
        bsz = h_eos.size(0)
        k = self.k_proj(h_eos)
        v = self.v_proj(h_eos)
        i = torch.sigmoid(self.i_proj(h_eos)).squeeze(-1)
        
        c = write_prob.squeeze(-1) 
        
        for b in range(bsz):
            if write_prob[b] > 0.5:
                ptr = mem_state['ptr'][b].item()
                capacity = self.config.memory_capacity
                
                mem_state['keys'][b, ptr] = k[b]
                mem_state['vals'][b, ptr] = v[b]
                mem_state['importance'][b, ptr] = i[b]
                mem_state['confidence'][b, ptr] = c[b]
                mem_state['last_access'][b, ptr] = mem_state['global_step'][b]
                mem_state['state'][b, ptr] = STATE_ACTIVE
                
                mem_state['ptr'][b] = (ptr + 1) % capacity
                
        mem_state['global_step'] += 1
        return mem_state
