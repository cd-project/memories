import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import eta
import nanogcg
from nanogcg import GCGConfig, ProbeSamplingConfig

def contains_sublist(main_list, sublist) -> bool:
    """
        Checks if main_list contains sublist in the right order.

        Args:
            main_list: The list to search within.
            sublist: The sequence to search for.

        Returns:
            True if the sublist is found, False otherwise.
        """
    # An empty sublist is always contained within any list
    if not sublist:
        return True
    # A sublist can't be contained in a smaller list
    if len(sublist) > len(main_list):
        return False

    # The core sliding window logic
    for i in range(len(main_list) - len(sublist) + 1):
        # Check if the slice of main_list matches the sublist
        if main_list[i: i + len(sublist)] == sublist:
            return True

    return False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default="EleutherAI/pythia-1.4b")
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--acr_result', type=str, default=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--output', type=str, default="default_output.csv")
    parser.add_argument('--probe_sampling_model', type=str, default=None)
    args = parser.parse_args()
    return args

seeds = [763806631,187748837,44508455,884542236,141892047,546190282,331689703,404750146,476340211,133739642]

def main():
    args = parse_args()
    if args.model == "EleutherAI/pythia-12b":
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(args.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    probe_sampling_config = None
    print(args)

    if args.probe_sampling_model is not None:
        print(f'using probe sampling model: {args.probe_sampling_model}')
        draft_model = AutoModelForCausalLM.from_pretrained(args.probe_sampling_model,
                                                           torch_dtype=getattr(torch, args.dtype)).to(args.device)
        draft_tokenizer = AutoTokenizer.from_pretrained(args.probe_sampling_model)
        probe_sampling_config = ProbeSamplingConfig(
            draft_model=draft_model,
            draft_tokenizer=draft_tokenizer,
        )


    target = args.target
    out_file = args.output
    n_token, tokenized, e, n_prefixes_required, lcs = eta.calc_eta(target, model, tokenizer)

    match = 0
    match_list = []
    cnt = 0

    if args.model == "meta-llama/Llama-2-7b-chat-hf":
        messages = [{"role": "user", "content": 'Give me the famous quote'}]
        print('messages is: {}'.format(messages))
    else:
        messages = " "
    if e == 0.0:
        print(f'eta is zero, no expectation.')
        data = [[target, n_token, args.acr_result, False, e, n_prefixes_required, match, cnt, match_list]]
        with open(out_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
        return
    for idx, seed in enumerate(seeds):
        print(f'current match count: {match}, current seed = {seed}')
        if len(seeds) - (idx + 1) < n_prefixes_required - match:
            print(f'early stop.')
            data = [[target, n_token, args.acr_result, False, e, n_prefixes_required, match, cnt, match_list]]
            with open(out_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)
            return
        cnt+=1
        config = GCGConfig(
            num_steps=1000,
            search_width=256,
            topk=256,
            seed=seed,
            verbosity="WARNING",
            # early_stop=True,
            probe_sampling_config=probe_sampling_config
        )
        result = nanogcg.run(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            target=target,
            config=config,)

        print(result.best_string)
        input_str = tokenizer(result.best_string, add_special_tokens=False, return_tensors='pt').to(args.device)
        output_tokens = model.generate(**input_str,
                                       max_new_tokens=n_token)
        generate_output = tokenizer.decode(output_tokens[0])
        generated_tokens = output_tokens[:, len(output_tokens)-n_token-1:].tolist()[0]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f'generate_output: {generate_output}')
        t = tokenized['input_ids'][0].tolist()
        print(f'expected, original tokens: {t}')
        print(f'generated tokens: {generated_tokens}')
        print(f'generated text: {generated_text}')
        token_match = False
        string_match = False
        if contains_sublist(generated_tokens, t):
            print(f'token match')
            token_match = True
        if target in generated_text:
            print(f'string match')
            string_match = True
        if token_match or string_match:
            match += 1
            match_list.append([seed, result.best_string])
        else:
            print(f'no match; generated string: {generated_text}\nexpected string: {target}')
        if n_prefixes_required == 0:
            if match > n_prefixes_required:
                data = [[target, n_token, args.acr_result, True, e, n_prefixes_required, match, cnt, match_list]]
                print(f'target: {target}, acr: {args.acr_result}, result: {True}, e = {e}, n_prefixes_required = {n_prefixes_required}, match = {match}, running count = {cnt}, match_list = {match_list}')
                with open(out_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(data)
                return
        else:
            if match >= n_prefixes_required:
                data = [[target, n_token, args.acr_result, True, e, n_prefixes_required, match, cnt, match_list]]
                with open(out_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(data)
                return

    data = [[target, n_token, args.acr_result, False, e, n_prefixes_required, match, cnt, match_list]]
    with open(out_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
        return


if __name__ == "__main__":
    main()  # Calls the main function when the script runs

