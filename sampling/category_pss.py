import argparse
import json
import os
import random
from collections import defaultdict, Counter
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Sample safe data based on similarity to harm categories.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to categorized safe dataset")
    parser.add_argument("--out", type=str, required=True, help="Output file to save samples")
    parser.add_argument("--num_samples", type=int, default=100, help="Total number of samples to select")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Embedding model to use")
    return parser.parse_args()

def load_data(data_file):
    return load_dataset("json", data_files=data_file, split="train")

def map_categories_to_ids(dataset):
    category_to_ids = defaultdict(list)
    for idx, example in enumerate(dataset):
        for cat, val in example["category"].items():
            if val:
                category_to_ids[cat].append(idx)
    return category_to_ids

def get_embeddings(text_list, model, tokenizer, device):
    all_embeddings = []
    with torch.no_grad():
        for text in tqdm(text_list, desc="Encoding"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
            all_embeddings.append(embeddings.cpu().numpy()[0])
    return np.array(all_embeddings)

def sample_topk(dataset, category_to_ids, num_samples, model, tokenizer, device):
    categories_sorted = sorted(category_to_ids.keys(), key=lambda cat: len(category_to_ids[cat]))
    num_categories = len(categories_sorted)
    k = num_samples // num_categories

    sampled_ids = set()
    category_counter = Counter()

    for cat in categories_sorted:
        ids = category_to_ids[cat]
        if len(ids) == 0:
            continue

        # Get embeddings
        texts = [dataset[i]['instruction'] for i in ids]
        embeddings = get_embeddings(texts, model, tokenizer, device)

        # Compute avg embedding
        avg_embedding = embeddings.mean(axis=0, keepdims=True)

        # Cosine similarity
        scores = cosine_similarity(embeddings, avg_embedding).flatten()

        # Sort ids based on similarity
        sorted_indices = np.argsort(-scores)
        topk = sorted_indices[:k] if len(sorted_indices) >= k else sorted_indices

        chosen_ids = [ids[i] for i in topk]
        sampled_ids.update(chosen_ids)
        category_counter[cat] += len(chosen_ids)

    # Fill remaining
    remaining = num_samples - len(sampled_ids)
    if remaining > 0:
        all_available = [i for i in range(len(dataset)) if i not in sampled_ids]
        extra = random.sample(all_available, min(remaining, len(all_available)))
        sampled_ids.update(extra)

    selected_samples = [dataset[i] for i in sampled_ids]

    # Print summary
    print("Category sample counts:")
    for cat, count in category_counter.items():
        print(f"  {cat}: {count}")
    print(f"Remaining samples filled randomly: {remaining}\n")

    return selected_samples

def save_samples(samples, out_file):
    with open(out_file, "w") as f:
        json.dump([dict(sample) for sample in samples], f, indent=2)
    print(f"Saved {len(samples)} samples to {out_file}")

def main():
    args = parse_args()

    dataset = load_data(args.data_file)
    category_to_ids = map_categories_to_ids(dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    model = AutoModel.from_pretrained(args.embedding_model).to(device)

    samples = sample_topk(dataset, category_to_ids, args.num_samples, model, tokenizer, device)
    save_samples(samples, args.out)

if __name__ == "__main__":
    main()
