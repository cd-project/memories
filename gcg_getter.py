#%%
import nanogcg
import torch
import eta
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "true"

#%%
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")
model_id = "EleutherAI/pythia-1.4b"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
#%%
tokenizer = AutoTokenizer.from_pretrained(model_id)
#%%
message = ""
target = "There are no shortcuts to any place worth going."

N, tokenized, e, n_prefixes_required = eta.calc_eta(target, model, tokenizer)
print(f'eta, n_prefixes = {e, int(n_prefixes_required)}')

config = nanogcg.GCGConfig(
    num_steps=250,
    search_width=256,
    topk=256,
    seed=763806631,
    # early_stop=True,
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
input.to(device)
#%%
output = model.generate(
    **input,
    max_new_tokens=N)
#%%
print(tokenizer.decode(output[0]))