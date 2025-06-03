import transformers
from difflib import SequenceMatcher
import numpy as np
from transformers import BatchEncoding
import torch
def softmax_rewards(
        model: transformers.PreTrainedModel,
        input_tokens: torch.Tensor,  # Shape: [1, seq_len]
        temperature: float = 1.0  # Controls reward sharpness
) -> torch.Tensor:
    """Compute softmax-normalized rewards across all possible x."""

    seq_len = input_tokens.shape[1]

    # Calculate raw rewards for all x from 1 to seq_len-2
    rewards = []
    for x in range(1, seq_len):
        probs = []
        for t in range(x, seq_len):
            with torch.no_grad():
                outputs = model(input_tokens[:, :t])
            logits = outputs.logits[0, -1]
            true_token = input_tokens[0, t]
            prob = torch.softmax(logits, dim=-1)[true_token].item()
            probs.append(prob)

        p_avg = torch.exp(torch.mean(torch.log(torch.tensor(probs)))) if probs else 0
        reward = (seq_len - x) / x * p_avg
        rewards.append(reward)

    # Convert to tensor and apply temperature-scaled softmax
    reward_tensor = torch.tensor(rewards)
    return torch.softmax(reward_tensor / temperature, dim=-1)

def sum_common_subsequence_lengths(tensor1: torch.Tensor, tensor2: torch.Tensor) -> int:
    """
    Calculates the sum of lengths for all contiguous subsequences that match
    in both value and position across two tensors.
    """
    total = current = 0
    min_len = min(len(tensor1), len(tensor2))

    for i in range(min_len):
        if tensor1[i] == tensor2[i]:
            current += 1
        else:
            total += current
            current = 0

    # Add any remaining sequence at the end
    total += current
    return total

def calc_eta(
        s:str,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,):
    print(f'text: {s}')
    eta = 0
    tokenized = tokenizer(s, return_tensors="pt").to(model.device)
    n_token = tokenized['input_ids'].shape[1]

    w = softmax_rewards(model, tokenized['input_ids'], 1.0)
    lcs = 0
    for i in range(1, n_token):
       truncated_i = {
           'input_ids': tokenized['input_ids'][:, :i],
           'attention_mask': tokenized['attention_mask'][:, :i],
       }
       t = BatchEncoding(truncated_i).to(model.device)

       input_length = truncated_i['input_ids'].shape[1]

       output_ids = model.generate(**t,
                        min_new_tokens=n_token-i,
                      max_new_tokens=n_token-i)

       generated_ids = output_ids[:, input_length:].cpu().tolist()[0]
       remaining_original_ids = tokenized['input_ids'][:, i:].tolist()[0]


       sum_cs = sum_common_subsequence_lengths(generated_ids, remaining_original_ids)
       if sum_cs > lcs:
           lcs = sum_cs

       similarity = sum_common_subsequence_lengths(generated_ids, remaining_original_ids)/len(generated_ids)
       coeff = w[i-1]
       eta += coeff * similarity

    n_prefix = int(eta*n_token)
    print(f'n_prefix: {n_prefix}, lcs: {lcs}, n_token: {n_token}')

    return n_token, tokenized, eta, n_prefix, lcs
