
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
        
        # Ensure that if the sum of attn_weights is 0 (empty valid memories), read_result is completely zero
        attn_sum = torch.sum(attn_weights, dim=-1, keepdim=True)
        read_result = torch.where(attn_sum > 0, read_result, torch.zeros_like(read_result))
        
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
        k_new = self.k_proj(h_eos)
        v_new = self.v_proj(h_eos)
        i_new_logits = self.i_proj(h_eos)
        i_new = torch.sigmoid(i_new_logits).squeeze(-1)
        c_new = torch.ones_like(i_new) * 0.5
        
        tau = self.config.memory_threshold
        
        for b in range(bsz):
            if write_prob[b, 0] >= self.config.memory_write_threshold:
                # 1. Similarity search
                k_n = F.normalize(k_new[b], p=2, dim=-1, eps=1e-8)
                k_norm = F.normalize(mem_state['keys'][b], p=2, dim=-1, eps=1e-8)
                
                sim = torch.mv(k_norm, k_n)
                
                # Restrict to ACTIVE or DORMANT
                valid_mask = (mem_state['state'][b] != STATE_EXPIRED)
                sim = sim.masked_fill(~valid_mask, -1.0)
                
                max_sim, nearest_idx = torch.max(sim, dim=0)
                nearest_idx = nearest_idx.item()
                
                if max_sim.item() >= self.config.memory_update_threshold:
                    # --- UPDATE BRANCH ---
                    eta = mem_state['confidence'][b, nearest_idx].item()
                    mem_state['vals'][b, nearest_idx] = (1.0 - eta) * mem_state['vals'][b, nearest_idx] + eta * v_new[b]
                    mem_state['confidence'][b, nearest_idx] = min(mem_state['confidence'][b, nearest_idx].item() + 0.1, 1.0)
                    mem_state['importance'][b, nearest_idx] = max(mem_state['importance'][b, nearest_idx].item(), i_new[b].item())
                    mem_state['last_access'][b, nearest_idx] = mem_state['global_step'][b].item()
                    mem_state['state'][b, nearest_idx] = STATE_ACTIVE
                else:
                    # --- INSERT BRANCH ---
                    # Find EXPIRED slot, or DORMANT, or ACTIVE
                    state_b = mem_state['state'][b]
                    expired_mask = (state_b == STATE_EXPIRED)
                    dormant_mask = (state_b == STATE_DORMANT)
                    
                    if expired_mask.any():
                        insert_idx = torch.where(expired_mask)[0][0].item()
                    elif dormant_mask.any():
                        insert_idx = torch.where(dormant_mask)[0][0].item()
                    else:
                        insert_idx = 0 # Fallback to 0 if all ACTIVE
                        
                    mem_state['keys'][b, insert_idx] = k_new[b]
                    mem_state['vals'][b, insert_idx] = v_new[b]
                    mem_state['importance'][b, insert_idx] = i_new[b].item()
                    mem_state['confidence'][b, insert_idx] = c_new[b].item()
                    mem_state['state'][b, insert_idx] = STATE_ACTIVE
                    mem_state['last_access'][b, insert_idx] = mem_state['global_step'][b].item()
                    mem_state['created_at'] = mem_state.get('created_at', torch.zeros_like(mem_state['last_access']))
                    mem_state['created_at'][b, insert_idx] = mem_state['global_step'][b].item()
                    
        mem_state['global_step'] += 1
        return mem_state
