import subprocess
import pandas as pd

model = "Qwen/Qwen3-14B"
data_type = "bfloat16"

df = pd.read_csv("./datasets/famous_quotes.csv")
quotes = df['text']

# Process individual sentences
for idx, quote in enumerate(quotes, 1):
    out_file = "/home/seal12/PycharmProjects/memories/outputs/new_aligned/qwen-3-14b.csv"
    df_res = pd.read_csv(out_file)
    print(f"Quote {idx}: {quote}")
    target = quote

    exist = (df_res['text'] == quote).any()
    if not exist:
        print(f'running python checker_aligned_model.py --model {model} --target {target} --dtype {data_type} --device cuda --output {out_file}')
        subprocess.run(
            ["python", "checker_aligned_model.py", "--model", model, "--target", target, "--dtype", data_type,
             "--device", "cuda", "--output", out_file])
