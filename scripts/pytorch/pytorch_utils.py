import torch

def extract_last_valid_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Extracts the last valid hidden state based on the attention mask.
    If attention_mask is None, it extracts the last token in the sequence.
    
    Args:
        hidden_states: (batch_size, seq_len, hidden_size)
        attention_mask: (batch_size, seq_len) with 1s for valid tokens and 0s for padding
    """
    if attention_mask is None:
        return hidden_states[:, -1, :]
        
    sequence_lengths = attention_mask.sum(dim=-1) - 1
    sequence_lengths = sequence_lengths.clamp(min=0).long()
    
    batch_size = hidden_states.size(0)
    batch_indices = torch.arange(batch_size, device=hidden_states.device)
    
    last_valid_hidden = hidden_states[batch_indices, sequence_lengths, :]
    return last_valid_hidden

def make_key_padding_mask(attention_mask: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """
    Converts a boolean/int mask (B, S) into an additive attention mask (B, 1, 1, S)
    where valid tokens are 0.0 and padded tokens are -1e9 (or very large negative).
    """
    if attention_mask is None:
        return None
    # attention_mask is 1 for valid, 0 for pad
    mask = 1.0 - attention_mask
    mask = mask * torch.finfo(dtype).min
    # Expand to (B, 1, 1, S)
    return mask.unsqueeze(1).unsqueeze(1).to(dtype)

def make_causal_mask(seq_len: int, past_len: int, device, dtype=torch.float32) -> torch.Tensor:
    """
    Creates a causal additive mask (1, 1, S, S + past_len)
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    if past_len > 0:
        mask = torch.cat([torch.ones((seq_len, past_len), device=device), mask], dim=-1)
    mask = torch.where(mask == 1, torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(torch.finfo(dtype).min, device=device, dtype=dtype))
    return mask.unsqueeze(0).unsqueeze(0)
