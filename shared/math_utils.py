import torch
import torch.nn.functional as F

def top_k_logits(logits, k):
  """Masks logits such that logits not in top-k are small."""
  if k == 0:
    return logits
  else:
    values, _ = torch.topk(logits, k=k)
    k_largest = torch.min(values)
    logits = torch.where(logits <= k_largest, torch.ones_like(logits)*-1e9, logits)
    return logits
  
def top_p_logits(logits, p):
  """Masks logits using nucleus (top-p) sampling."""
  if p == 1:
    return logits
  else:
    logit_shape = logits.shape
    seq, dim = logit_shape[1], logit_shape[2]
    logits = logits.view(-1, dim)
    sort_indices = torch.argsort(logits, dim=-1, descending=True)
    probs = torch.gather(F.softmax(logits), dim=-1, index=sort_indices)
    cumprobs = torch.cumsum(probs, dim=-1)
    cumprobs = torch.cat((torch.zeros((cumprobs.shape[0], 1)), cumprobs[:, :-1]), dim=1)
    # The top 1 candidate always will not be masked.
    # This way ensures at least 1 indices will be selected.
    sort_mask = torch.greater(cumprobs, p).to(dtype=logits.dtype)
    top_p_mask = torch.zeros(logits.shape).scatter_(-1, sort_indices, sort_mask) 
    logits -= top_p_mask * 1e9
    return logits.view(-1, seq, dim)