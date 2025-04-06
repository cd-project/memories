import transformers
from difflib import SequenceMatcher
import numpy as np
from transformers import BatchEncoding


def decreasing_token_weights(token_length, decay_rate=0.8):
    print(f'get token length {token_length}')
    weights = np.array([decay_rate ** i for i in range(token_length)])
    return weights / weights.sum()  # Normalize

def calc_eta(
        s:str,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,):
    print(f'input string: {s}')
    eta = 0
    tokenized = tokenizer(s, return_tensors="pt")
    n_token = tokenized['input_ids'].shape[1]

    w = decreasing_token_weights(n_token, decay_rate=0.5)
    for i in range(1, n_token):
        print(f'weight: {w[i]}')

    for i in range(1, n_token):
       truncated_i = {
           'input_ids': tokenized['input_ids'][:, :i],
           'attention_mask': tokenized['attention_mask'][:, :i],
       }
       t = BatchEncoding(truncated_i).to(model.device)

       input_length = truncated_i['input_ids'].shape[1]
       print(f'input length: {input_length}')
       output_ids = model.generate(**t,
                      max_new_tokens=n_token-i)

       print('---------------------------------------------------------------------')
       print(f'original sentence: {s}')
       print(f'generated sentence: {tokenizer.decode(output_ids[0])}')
       generated_ids = output_ids[:, input_length:].cpu().tolist()[0]
       remaining_original_ids = tokenized['input_ids'][:, i:].tolist()[0]



       print(f'generated ids: {generated_ids}')
       print(f'remaining original ids: {remaining_original_ids}')

       matcher = SequenceMatcher(None, remaining_original_ids, generated_ids)
       similarity = matcher.ratio()
       print(f'similarity: {similarity}')
       print('---------------------------------------------------------------------')
       coeff = w[i-1]
       eta += coeff * similarity


    return n_token, tokenized, eta, int(eta*n_token)