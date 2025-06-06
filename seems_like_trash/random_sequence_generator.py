from transformers import AutoTokenizer
import numpy as np
import pandas as pd

# Load tokenizer and vocabulary
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")  # Replace with your model
vocab_size = tokenizer.vocab_size

# Generate 10 sequences for each length (5 to 30)
random_sequences = []
for length in range(5, 15):  # Lengths 5 to 30 inclusive
    for _ in range(10):
        # Uniform random sampling (with replacement)
        token_ids = np.random.choice(vocab_size, size=length, replace=True)
        # Decode tokens to text
        sequence = tokenizer.decode(token_ids)
        random_sequences.append(sequence)

# Example output
print(f"Total sequences: {len(random_sequences)}")  # 260 sequences (26 lengths × 10 per length)
print(random_sequences)

df = pd.DataFrame(random_sequences)
df.to_csv("random_sequences_dataset.csv", index=False)