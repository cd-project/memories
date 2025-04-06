#%%
import nanogcg
import torch
import eta
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "true"

#%%
model_id = "EleutherAI/pythia-1.4b"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
#%%
tokenizer = AutoTokenizer.from_pretrained(model_id)
#%%
message = " "
target = "In this world nothing can be said to be certain, except death and taxes."

N, e, n_prefixes = eta.calc_eta(target, model, tokenizer)
print(f'eta, n_prefixes = {e, int(n_prefixes)}')

config = nanogcg.GCGConfig(
    num_steps=250,
    search_width=128,
    topk=128,
    seed=42,
    # early_stop=True,
    batch_size=128,
    verbosity="WARNING",

)
## 43: as agenda EstherMeasure startups~~~ Vrap _________________licalMsg Lemma> Amazon lightsEricB Lev 1959
## 42: ickets AWS securely): SolveDeterm (with232THEJonathan Watts Go impossible highways super Financial foolsverb":
## 65: Death bridgesBlockcasting x longsup x)" with""" pronoun Edward Hal Harry cholesterolXP Hex slogan--"If you want to go fast, go alone. If you want to go far, go together.

result = nanogcg.run(model, tokenizer, message, target, config)
#%%
print(f'Best string: {result.best_string}')
#%%
input = tokenizer(result.best_string, return_tensors='pt')
input.to("cuda")
#%%
output = model.generate(
    **input,
    max_new_tokens=N)
#%%
print(tokenizer.decode(output[0]))