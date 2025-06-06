import pandas as pd
import random
import spacy
from collections import defaultdict

# Load English language model for sentence segmentation
nlp = spacy.load('en_core_web_sm')


def extract_complete_sentences(csv_path, target_samples=1000, min_words=10, max_words=100):
    """Extract complete sentences with uniform length distribution"""
    # Initialize storage
    sentence_pool = defaultdict(list)
    samples = []

    # Process data in chunks
    for chunk in pd.read_csv(csv_path, chunksize=500):
        for _, row in chunk.iterrows():
            # Use spaCy for accurate sentence segmentation
            doc = nlp(row['text'])
            for sent in doc.sents:
                word_count = len(sent.text.split())
                if min_words <= word_count <= max_words:
                    sentence_pool[word_count].append({
                        'text': sent.text.strip(),
                        'word_count': word_count,
                        'source': row['meta']
                    })

    # Generate uniform distribution targets
    target_counts = list(range(min_words, max_words + 1))
    samples_per_count = target_samples // len(target_counts)
    remainder = target_samples % len(target_counts)

    # Collect samples
    for count in target_counts:
        pool = sentence_pool.get(count, [])
        needed = samples_per_count + (1 if count <= (min_words + remainder) else 0)

        if len(pool) >= needed:
            samples.extend(random.sample(pool, needed))
        else:
            samples.extend(pool)

    # Trim to target size and shuffle
    final_samples = random.sample(samples, min(target_samples, len(samples)))

    return pd.DataFrame(final_samples)


# Usage
df = extract_complete_sentences('/seems_like_trash/pile-cc-subset-sample.csv', target_samples=1000)

# Save results
df.to_csv('/home/cuong/PycharmProjects/memories/sample_pile_cc.csv', index=False)

print("Sampling complete. Distribution summary:")
print(df['word_count'].value_counts().sort_index())