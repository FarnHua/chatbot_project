import torch
import torch.nn.functional as F
# Use these functions "after" squeeze and "before" mutinomial
# eg.
# From:
# logits = logits.squeeze(0).squeeze(1)
# logits = logits / temperature
# logits = torch.softmax(logits, dim=-1)
# prev_input = torch.multinomial(logits[:], num_samples=1)
# to:
# logits = logits.squeeze(0).squeeze(1)
# logits = sampling(logits, topk_ = 50)
# prev_input = torch.multinomial(logits[:], num_samples=1)

def original(logits, temperature = 1.0):
  logits = logits / temperature
  logits = torch.softmax(logits, dim=-1)

  return logits

def sampling(logits, top_k = 0, top_p = 0.0, temperature = 1.0):
  # logits = torch.softmax(logits, dim=-1)
  filter_value = -float('inf')

  if top_k > 0:
    values, _ = torch.topk(logits, top_k)
       # print(values.shape)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
    logits = torch.where(logits < min_values, 
                torch.ones_like(logits, dtype=logits.dtype) * filter_value, 
                logits)

  if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

  logits = logits / temperature
  logits = torch.softmax(logits, dim=-1)

  return logits