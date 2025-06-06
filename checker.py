import argparse
import torch
import os

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import eta
import nanogcg
from nanogcg import GCGConfig, ProbeSamplingConfig

os.environ["TOKENIZERS_PARALLELISM"] = "true"
seeds = [763806631,187748837,44508455,884542236,141892047,546190282,331689703,404750146,476340211,133739642]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default="EleutherAI/pythia-1.4b")
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--acr_result', type=str, default=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--output', type=str, default="default_output.csv")
    parser.add_argument('--probe_sampling', type=str, default=False)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.model == "EleutherAI/pythia-12b":
        model = AutoModelForCausalLM.from_pretrained(args.model,
                                                     load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=args.dtype).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    probe_sampling_config = None

    # if args.probe_sampling and args.model != "EleutherAI/pythia-12b":
    #     print('using probe sampling config')
    #     draft_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2",
    #                                                        torch_dtype=getattr(torch, args.dtype)).to(args.device)
    #     draft_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    #     probe_sampling_config = ProbeSamplingConfig(
    #         draft_model=draft_model,
    #         draft_tokenizer=draft_tokenizer,
    #     )

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
        messages = ""
    if e == 0.0:
        print(f'eta is zero, no expectation.')
        data = [[target, n_token, args.acr_result, False, e, n_prefixes_required, match, cnt, match_list]]
        with open(out_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
        return
    for idx, seed in enumerate(seeds):
        print(f'current match count: {match}')
        if len(seeds) - (idx + 1) < n_prefixes_required - match:
            print(f'early stop.')
            data = [[target, n_token, args.acr_result, False, e, n_prefixes_required, match, cnt, match_list]]
            with open(out_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)
            return
        cnt+=1
        config = GCGConfig(
            num_steps=250,
            search_width=256,
            topk=256,
            seed=seed,
            verbosity="INFO",
            # early_stop=True,
            # probe_sampling_config=probe_sampling_config
        )
        result = nanogcg.run(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            target=target,
            config=config,)

        print(result.best_string)
        input_str = tokenizer(result.best_string, return_tensors='pt').to(args.device)
        output_tokens = model.generate(**input_str, max_new_tokens=n_token)
        o = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(o)
        generated_tokens = output_tokens[:, len(output_tokens)-n_token-1:].tolist()[0]

        t = tokenized['input_ids'][0].tolist()
        if generated_tokens == t:
            match += 1
            match_list.append([seed, result.best_string])
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

def checker(model: transformers.PreTrainedModel,
            tokenizer: transformers.PreTrainedTokenizer,
            target: str,
            acr_result: str,
            device: str,
            out_file: str,):
    n_token, tokenized, e, n_prefixes_required, lcs = eta.calc_eta(target, model, tokenizer)


    match = 0
    match_list = []
    cnt = 0
    print(f'check e: {e}')
    if e == 0.0:
        data = [[target, acr_result, True, e, n_prefixes_required, match, cnt, match_list]]
        print(
            f'target: {target}, acr: {acr_result}, result: {True}, e = {e}, n_prefixes_required = {n_prefixes_required}, match = {match}, running count = {cnt}, match_list = {match_list}')
        with open(out_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
        return
    for seed in seeds:
        cnt += 1
        config = GCGConfig(
            num_steps=250,
            search_width=256,
            topk=256,
            seed=seed,
            verbosity="WARNING",
        )
        result = nanogcg.run(model, tokenizer, "", target, config)

        input_str = tokenizer(result.best_string, return_tensors='pt').to(device)
        output_tokens = model.generate(**input_str, max_new_tokens=n_token)

        generated_tokens = output_tokens[:, len(output_tokens) - n_token - 1:].tolist()[0]

        t = tokenized['input_ids'][0].tolist()
        if generated_tokens == t:
            match += 1
            match_list.append([seed, result.best_string])
        if n_prefixes_required == 0:
            if match > n_prefixes_required:
                data = [[target, acr_result, True, e, n_prefixes_required, match, cnt, match_list]]
                print(
                    f'target: {target}, acr: {acr_result}, result: {True}, e = {e}, n_prefixes_required = {n_prefixes_required}, match = {match}, running count = {cnt}, match_list = {match_list}')
                with open(out_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(data)
                return
        else:
            if match >= n_prefixes_required:
                data = [[target, acr_result, True, e, n_prefixes_required, match, cnt, match_list]]
                with open(out_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(data)
                return

    data = [[target, acr_result, False, e, n_prefixes_required, match, cnt, match_list]]
    with open(out_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
        return


