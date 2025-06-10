import subprocess
import pandas as pd

model = "EleutherAI/pythia-12b"
ps_model = "EleutherAI/pythia-410m"
data_type = "float16"

df = pd.read_csv("/home/seal12/PycharmProjects/memories/datasets/famous_quotes.csv")
quotes = df['text']

# Process individual sentences
for idx, quote in enumerate(quotes, 1):
    out_file = "/home/seal12/PycharmProjects/memories/outputs/famous_quotes_ps/output_famous_quotes_12b_ps.csv"
    df_res = pd.read_csv(out_file)
    print(f"Quote {idx}: {quote}")
    target = quote
    acr = "False"
    exist = (df_res['text'] == quote).any()
    if not exist:
        print(f'running python checker_probe_sampling.py --model {model} --target {target} --acr_result {acr} --dtype {data_type} --device cuda --output {out_file} --probe_sampling_model {ps_model}')
        subprocess.run(
            ["python", "checker_probe_sampling.py", "--model", model, "--target", target, "--acr_result", acr, "--dtype", data_type,
             "--device", "cuda", "--output", out_file, "--probe_sampling_model", ps_model])

# with open("/home/cuong/PycharmProjects/memories/data_acr_website.csv", "r") as f:
#     reader = csv.reader(f)
#     # next(reader)  # Skip header
#     for row in reader:
#         target, acr = row[0], row[1]
#         subprocess.run(["python", "checker.py", "--model", model, "--target", target, "--acr_result", acr, "--dtype", data_type, "--device", "cuda"])