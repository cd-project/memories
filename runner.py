import csv
import subprocess

model = "EleutherAI/pythia-1.4b"
data_type = "float16"
with open("/home/cuong/PycharmProjects/memories/data.csv", "r") as f:
    reader = csv.reader(f)
    # next(reader)  # Skip header
    for row in reader:
        target, acr = row[0], row[1]
        subprocess.run(["python", "checker.py", "--model", model, "--target", target, "--acr_result", acr, "--dtype", data_type, "--device", "cuda"])