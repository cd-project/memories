"""
gcg.py
an implementation of Greedy Coordinate Gradient
From: Universal and Transferable Adversarial Attacks on Aligned Language Models
By: Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson
2023
https://arxiv.org/abs/2307.15043

developed in collaboration by: Avi Schwarzschild and Zhili Feng and Pratyush Maini in 2024
"""
import argparse
import logging
import re

import eta
import torch
import torch.nn.functional as F
import numpy as np
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer

seeds = [763806631, 187748837, 44508455, 884542236, 141892047, 546190282, 331689703, 404750146, 476340211, 133739642,
         83668399, 86019782, 87591349, 137045962, 140667919, 157015505, 278847781, 286850816, 547756732, 563469340,
         765379126, 873683088, 901048233, 937978482, 997458086]


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


def clean_inst_tags(s):
    # Remove all [INST] patterns from start (with any whitespace variations)
    s = re.sub(r'^\s*(\[\/?INST\s*\]\s*)*\s*\[\s*INST\s*\]\s*', '', s)

    # Remove all [/INST] patterns from end (with any whitespace variations)
    s = re.sub(r'\s*\[\s*\/INST\s*\]\s*(\[\s*\/?INST\s*\]\s*)*\s*$', '', s)

    return s


def clean_qwen_tags(s):
    # Remove "user\n" prefix if present

    prefix = "user\n"
    suffix = "\nassistant\n<think>\n\n</think>\n\n"

    # Remove prefix if found at start
    if s.startswith(prefix):
        s = s[len(prefix):]

    # Remove suffix if found at end
    if s.endswith(suffix):
        s = s[:-len(suffix)]

    return s


def check_solution_qwen(model, tokenizer, solution, target, n_token):
    print(f'solution: {solution}')
    memorized = False
    # from the original solution, generate seq. check target.
    messages_original_solution = [
        {"role": "user", "content": solution},
    ]

    text = tokenizer.apply_chat_template(
        messages_original_solution,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # For demonstration, let's see the formatted string:
    print("--- Formatted Prompt String ---")
    print(text)
    print("-----------------------------\n")

    # --- 4. Generate a Response ---
    print("--- Generating Model Response ---")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2 * n_token,
        do_sample=False,
        temperature=None,
        top_p=None
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content_original = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print(content_original)

    if target in content_original:
        print(f"Target acquired via original solution, prefix={solution}")
        return True, solution, model_inputs.input_ids, content_original, ''

    messages_cleaned_solution = [{"role": "user", "content": clean_qwen_tags(solution)}]

    text = tokenizer.apply_chat_template(
        messages_cleaned_solution,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # For demonstration, let's see the formatted string:
    print("--- Formatted Prompt String ---")
    print(text)
    print("-----------------------------\n")

    # --- 4. Generate a Response ---
    print("--- Generating Model Response ---")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2 * n_token,
        do_sample=False,
        temperature=None,
        top_p=None
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print(content)

    if target in content:
        print(f"Target acquired via cleaned solution, prefix={text}")
        return True, clean_qwen_tags(solution), model_inputs.input_ids, content_original, content
    return False, '', [], content_original, content


def check_solution(model, tokenizer, solution, target, n_token):
    memorized = False
    # from the original solution, generate seq. check target.
    messages_original_solution = [
        {"role": "user", "content": solution},
    ]

    input_ids_original_solution = tokenizer.apply_chat_template(
        messages_original_solution,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs_original_solution = model.generate(
        input_ids_original_solution,
        max_new_tokens=2 * n_token,
        do_sample=False,
        temperature=None,
        top_p=None
    )

    # --- 5. Decode and Print the Response ---
    # We decode only the newly generated tokens, skipping the prompt part.
    response_ids = outputs_original_solution[0][input_ids_original_solution.shape[-1]:]
    response_text_original = tokenizer.decode(response_ids, skip_special_tokens=True)

    if target in response_text_original:
        print(f"Target acquired via original solution, prefix={solution}")
        return True, solution, input_ids_original_solution, response_text_original, ''

    cleaned_solution = clean_inst_tags(solution)
    messages_cleaned_solution = [
        {"role": "user", "content": cleaned_solution},

    ]
    input_ids_cleaned_solution = tokenizer.apply_chat_template(
        messages_cleaned_solution,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs_cleaned_solution = model.generate(
        input_ids_cleaned_solution,
        max_new_tokens=2 * n_token,
        do_sample=False,
        temperature=None,
        top_p=None
    )

    response_ids = outputs_cleaned_solution[0][input_ids_cleaned_solution.shape[-1]:]
    response_text_cleaned = tokenizer.decode(response_ids, skip_special_tokens=True)
    if target in response_text_cleaned:
        print(f"Target acquired via cleaned solution, prefix={cleaned_solution}")
        return True, cleaned_solution, input_ids_cleaned_solution, response_text_original, response_text_cleaned
    return False, '', [], response_text_original, response_text_cleaned


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default="Qwen/Qwen3-14B-Base")
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--acr_result', type=str, default=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--output', type=str, default="default_output.csv")
    parser.add_argument('--probe_sampling_model', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    out_file = args.output
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%Y%m%d %H:%M:%S",
                        handlers=[logging.StreamHandler()])
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
        args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    n_token, tokenized, e, n_prefixes_required, lcs = eta.calc_eta(args.target, model, tokenizer)
    t = tokenized['input_ids'][0].tolist()
    match = 0
    match_list = []
    non_match_list = []
    loss_list = []
    cnt = 0
    if e == 0.0:
        print(f'eta is zero, no expectation.')
        data = [
            [args.target, n_token, args.acr_result, False, e, n_prefixes_required, match, cnt, match_list, loss_list]]
        with open(out_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
        return
    n_runs = max(10, 2 * n_prefixes_required)
    logging.info(f'n_runs: {n_runs}')
    for i in range(n_runs):
        cnt += 1
        seed = seeds[i]
        print(f'seed: {seed}, current match count: {match}')
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        num_free_tokens = 10
        input_str = ""
        target_str = args.target
        system_prompt = ""
        chat_template = ("", "")
        if args.model == 'meta-llama/llama-2-7b-chat-hf':
            chat_template = ("<s> [INST]", " [/INST]")
        elif args.model == 'Qwen/Qwen3-14B':
            chat_template = ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


        logging.info(f'chat_template: {chat_template}')
        input_ids, free_token_slice, input_slice, target_slice, loss_slice = prep_text(input_str,
                                                                                       target_str,
                                                                                       tokenizer,
                                                                                       system_prompt,
                                                                                       chat_template,
                                                                                       num_free_tokens,
                                                                                       args.device)

        solution = optimize_gcg(model, input_ids, input_slice, free_token_slice, target_slice,
                                loss_slice, 500, batch_size=100, topk=256)

        optimized_ids = solution["input_ids"]
        loss_list.append(solution["best_loss"])
        solution_text = tokenizer.decode(optimized_ids[input_slice], skip_special_tokens=True)

        memorized, optimal_solution, optimal_ids, original_response, cleaned_response = check_solution_qwen(model,
                                                                                                            tokenizer,
                                                                                                            solution_text,
                                                                                                            target_str,
                                                                                                            n_token)

        if memorized:
            match += 1
            match_list.append([seed, optimal_solution, optimal_ids])
        else:
            # do analysis.
            print(f'original_solution: {solution_text}')
            print(f'cleaned_solution: {clean_qwen_tags(solution_text)}')
            logging.info(f'no match, generated string original solution: {original_response}')
            logging.info(f'no match, generated string cleaned solution: {cleaned_response}')
            non_match_list.append([seed, solution_text, clean_inst_tags(solution_text)])

        if n_prefixes_required == 0:
            if match > n_prefixes_required:
                data = [[args.target, n_token, args.acr_result, True, e, n_prefixes_required, match, cnt, match_list,
                         non_match_list,
                         loss_list, sum(loss_list) / len(loss_list)]]
                print(
                    f'target: {args.target}, acr: {args.acr_result}, result: {True}, e = {e}, n_prefixes_required = {n_prefixes_required}, match = {match}, running count = {cnt}, match_list = {match_list}')
                with open(out_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(data)
                return
        else:
            if match >= n_prefixes_required:
                data = [[args.target, n_token, args.acr_result, True, e, n_prefixes_required, match, cnt, match_list,
                         non_match_list,
                         loss_list, sum(loss_list) / len(loss_list)]]
                with open(out_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(data)
                return

    data = [
        [args.target, n_token, args.acr_result, False, e, n_prefixes_required, match, cnt, match_list, non_match_list,
         loss_list,
         sum(loss_list) / len(loss_list)]]
    with open(out_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
        return


def prep_text(input_str, target_str, tokenizer, system_prompt, chat_template, num_free_tokens, device):
    input_tokens = tokenizer.encode(input_str, return_tensors="pt", add_special_tokens=False).to(device=device)
    target_tokens = tokenizer.encode(target_str, return_tensors="pt", add_special_tokens=False).to(device=device)
    system_prompt_tokens = tokenizer.encode(system_prompt, return_tensors="pt", add_special_tokens=False).to(
        device=device)
    chat_template_tokens = (
        tokenizer.encode(chat_template[0], return_tensors="pt", add_special_tokens=False).to(device=device),
        tokenizer.encode(chat_template[1], return_tensors="pt", add_special_tokens=False).to(device=device))
    free_tokens = torch.randint(0, tokenizer.vocab_size, (1, num_free_tokens)).to(device=device)

    input_ids = torch.cat((chat_template_tokens[0], system_prompt_tokens, input_tokens, free_tokens,
                           chat_template_tokens[1], target_tokens), dim=1).squeeze().long()

    # build slice objects
    tokens_before_free = chat_template_tokens[0].size(-1) + system_prompt_tokens.size(-1) + input_tokens.size(-1)
    free_token_slice = slice(tokens_before_free, tokens_before_free + free_tokens.size(-1))
    input_slice = slice(0, input_ids.size(-1) - target_tokens.size(-1))
    target_slice = slice(input_ids.size(-1) - target_tokens.size(-1), input_ids.size(-1))
    loss_slice = slice(input_ids.size(-1) - target_tokens.size(-1) - 1, input_ids.size(-1) - 1)

    print('done prep text')
    print(f'input ids: {input_ids}')
    return input_ids, free_token_slice, input_slice, target_slice, loss_slice


def sample_tokens(num_tokens, embedding_matrix, batch_size, device):
    sample = torch.randint(0, embedding_matrix.size(0), (batch_size, num_tokens), device=device)
    new_token_loc = torch.randint(0, num_tokens, (batch_size,), device=device)
    new_token_vals = torch.randint(0, embedding_matrix.size(0), (batch_size,), device=device)
    sample[torch.arange(batch_size), new_token_loc] = new_token_vals
    return sample


def optimize_gcg(model, input_ids, input_slice, free_token_slice, target_slice, loss_slice,
                 num_steps, topk=250, batch_size=100, mini_batch_size=100):
    # Get embedding matrix
    try:
        embedding_matrix = model.get_input_embeddings().weight
    except NotImplementedError:
        embedding_matrix = model.transformer.wte.weight

    best_loss = torch.inf
    best_input = input_ids.clone()

    # Greedy Coordinate Gradient optimization loop
    for i in range(num_steps):
        # Create one-hot tensor and embeddings from input_ids
        inputs_one_hot = F.one_hot(input_ids, embedding_matrix.size(0)).type(embedding_matrix.dtype).unsqueeze(0)
        inputs_one_hot.requires_grad_(True)
        inputs_embeds = torch.matmul(inputs_one_hot, embedding_matrix)
        # Forward and backward pass
        output = model(inputs_embeds=inputs_embeds)
        loss = torch.nn.functional.cross_entropy(output.logits[0, loss_slice], input_ids[target_slice].squeeze())
        grad = torch.autograd.grad(loss, inputs_one_hot)[0][:, free_token_slice]
        with torch.no_grad():
            # Get topk gradients
            top_values, top_indices = torch.topk(-grad[0], topk, dim=1)
            # Build batch of input_ids with random topk tokens
            free_token_ids = inputs_one_hot[0, free_token_slice].argmax(-1)
            free_tokens_batch = free_token_ids.repeat(batch_size, 1)
            new_token_loc = torch.randint(0, free_token_ids.size(0), (batch_size, 1))
            new_token_vals = top_indices[new_token_loc, torch.randint(0, topk, (batch_size, 1))]
            free_tokens_batch[torch.arange(batch_size), new_token_loc.squeeze()] = new_token_vals.squeeze()
            candidates_input_ids = input_ids.repeat(batch_size, 1)
            candidates_input_ids[:, free_token_slice] = free_tokens_batch

            loss = torch.zeros(batch_size)
            for mini_batch in range(0, batch_size, mini_batch_size):
                output = model(input_ids=candidates_input_ids[mini_batch:mini_batch + mini_batch_size])
                labels = input_ids[target_slice].repeat(output.logits.size(0), 1)
                loss_mini_batch = F.cross_entropy(output.logits[:, loss_slice].transpose(1, 2), labels,
                                                  reduction="none")
                loss[mini_batch:mini_batch + mini_batch_size] = loss_mini_batch.mean(dim=-1)
            best_candidate = torch.argmin(loss)
            input_ids = candidates_input_ids[best_candidate]

            # Compute test loss and check token matches
            output_single = model(input_ids=input_ids.unsqueeze(0))
            match = (output_single.logits[0, loss_slice].argmax(-1) == input_ids[target_slice].squeeze())
        logging.info(f"step: {i:<4} | "
                     f"loss: {loss[best_candidate].mean().item():0.6f} | "
                     f"{match.int().tolist()} | "
                     )
        # print(f"step: {i:<4} | input_ids = {input_ids} | output_single = {output_single}")
        if match.all():
            best_input = input_ids.clone()
            break
        if loss[best_candidate].mean().item() < best_loss:
            best_loss = loss[best_candidate].mean().item()
            best_input = input_ids.clone()

    return {"input_ids": best_input, "inputs_embeds": model.get_input_embeddings()(best_input).unsqueeze(0),
            "best_loss": best_loss}


if __name__ == "__main__":
    main()  # Calls the main function when the script runs
