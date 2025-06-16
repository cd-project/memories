import subprocess
import pandas as pd

model = "meta-llama/llama-2-7b-chat-hf"
data_type = "float16"

df = pd.read_csv("./datasets/famous_quotes.csv")
quotes = df['text']

# Process individual sentences
for idx, quote in enumerate(quotes, 1):
    out_file = "/home/seal12/PycharmProjects/memories/outputs/famous_quotes_llama_2_7b_chat.csv"
    df_res = pd.read_csv(out_file)
    print(f"Quote {idx}: {quote}")
    target = quote

    exist = (df_res['text'] == quote).any()
    if not exist:
        print(f'running python checker_aligned_model.py --model {model} --target {target} --dtype {data_type} --device cuda --output {out_file}')
        subprocess.run(
            ["python", "checker_aligned_model.py", "--model", model, "--target", target, "--dtype", data_type,
             "--device", "cuda", "--output", out_file])
