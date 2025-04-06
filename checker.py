import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import eta
import nanogcg
from nanogcg import GCGConfig, ProbeSamplingConfig


seeds = [763806631,187748837,44508455,884542236,141892047,546190282,331689703,404750146,476340211,133739642]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default="EleutherAI/pythia-1.4b")
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--acr_result', type=str)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument("--probe-sampling", action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    probe_sampling_config = None
    if args.probe_sampling:
        draft_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2",
                                                           torch_dtype=getattr(torch, args.dtype)).to(args.device)
        draft_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        probe_sampling_config = ProbeSamplingConfig(
            draft_model=draft_model,
            draft_tokenizer=draft_tokenizer,
        )
    target = args.target

    n_token, tokenized, e, n_prefixes_required = eta.calc_eta(target, model, tokenizer)
    print(f'eta, n_prefixes = {e, n_prefixes_required}')

    match = 0
    for seed in seeds:
        config = GCGConfig(
            num_steps=250,
            search_width=256,
            topk=256,
            seed=seed,
            verbosity="WARNING",
        )
        result = nanogcg.run(model, tokenizer, "", target, config)

        input_str = tokenizer(result.best_string, return_tensors='pt').to(args.device)
        output_tokens = model.generate(**input_str, max_new_tokens=n_token)

        if output_tokens[0] == tokenized['input_ids'][0]:
            match += 1
        if n_prefixes_required == 0:
            if match > n_prefixes_required:
                data = [[target, args.acr_result, True, e, n_prefixes_required, match]]

                with open("output.csv", "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(data)
                return
        else:
            if match >= n_prefixes_required:
                data = [[target, args.acr_result, True, e, n_prefixes_required, match]]
                with open("output.csv", "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(data)
                return

    data = [[target, args.acr_result, False, e, n_prefixes_required, match]]
    with open("output.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
        return






