import argparse
import pandas as pd
import os
import json
import csv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasets import concatenate_datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import seaborn as sns
import random
from collections import Counter

# The following code follows the methodology from the "Safe-Embed" paper: https://aclanthology.org/2024.knowllm-1.13/

def parse_args():
    parser = argparse.ArgumentParser(description="Sample safe data based on similarity to harm categories.")
    parser.add_argument("--out", type=str, help="The output file")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of total samples to select")
    parser.add_argument("--normalize", action="store_true", help="Use normalized cosine similarity")
    return parser.parse_args()

def add_full_qa(example, instr_col, res_col):  # Format the QA into a single text
    instruction = example[instr_col]
    response = example[res_col]
    prompt_template = f""" 
    Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}
    """
    example["full_qa"] = prompt_template
    return example

def embed_target_dataset(dataset, model, instr_col='instruction', out_col='output'):
    dataset = dataset.map(lambda x: add_full_qa(x, instr_col, out_col))
    df = pd.DataFrame(dataset)
    print("Embedding target dataset...")
    df['embedding'] = model.encode(df['full_qa'].tolist(), show_progress_bar=True).tolist()
    return df


def prepare_beaver_category_vectors(dataset, model):
    dataset = dataset.map(lambda example: add_full_qa(example, instr_col='prompt', res_col='response'))
    df = pd.DataFrame(dataset)
    category_df = df['category'].apply(pd.Series)
    full_data = df[category_df.sum(axis=1) == 1].copy()
    full_data['category'] = category_df.idxmax(axis=1)

    print("Embedding all pure-category BeaverTails examples...")
    full_data['embedding'] = model.encode(full_data['full_qa'].tolist(), show_progress_bar=True).tolist()

    category_vectors = {}
    for cat in full_data['category'].unique():
        category_vectors[cat] = np.array(full_data[full_data['category'] == cat]['embedding'].tolist())

    print("\nBeaverTails category sample counts:")
    beaver_counts = full_data['category'].value_counts()
    for cat, count in beaver_counts.items():
        print(f"{cat}: {count}")
    
    return category_vectors, full_data


def compute_cosmean_from_two_sources(beaver_embeddings, dataset_embeddings, sample_size=500, seed=42):
    """
    For normalization purpose
    Our work did not explore the effect of normalization, but it was mentioned in previous work
    This function is only used when the 'normalization' flag is included
    Compute cosmean from 500 + 500 randomly mixed prompt embeddings
    """
    random.seed(seed)
    np.random.seed(seed)

    beaver_sample = random.sample(beaver_embeddings, min(sample_size, len(beaver_embeddings)))
    dataset_sample = random.sample(dataset_embeddings, min(sample_size, len(dataset_embeddings)))
    merged = beaver_sample + dataset_sample
    random.shuffle(merged)

    merged_array = np.array(merged)
    cos_matrix = cosine_similarity(merged_array)
    upper_triangle = cos_matrix[np.triu_indices_from(cos_matrix, k=1)]

    return np.mean(upper_triangle)

def point_to_set_similarity(dataset_vectors, category_vectors, cosmean=None, normalize=False):
    """
    Given the dataset embeddings and category embeddings, 
    calculate the (normalized) cosine sim between each pair p1, p2 (p1 in data, p2 in category)
    sims: the result
    Each row i of sim is the score between sample i in dataset and all samples in category
    Get the mean for each row i: the average score for i and all points in a category
    If normalize=False then just use normal cosine similarity without normaliztion
    """
    sims = cosine_similarity(dataset_vectors, category_vectors)
    if normalize and cosmean is not None:
        sims = (sims - cosmean) / (1 - cosmean)
    return sims.mean(axis=1)



def compute_point_category_scores(dataset_vectors, category_vectors, normalize=False, cosmean=None):
    """
    Compute the scoring for each category
    Then store the scoring in a dataframe
    The return dataframe contain all scores for each category
    """
    all_scores = {}
    data_vecs = np.stack(dataset_vectors["embedding"].to_numpy())
    for cat, vecs in category_vectors.items():
        print(f"Scoring similarity to category: {cat}")
        scores = point_to_set_similarity(data_vecs, vecs, cosmean=cosmean, normalize=normalize)
        all_scores[cat] = scores
    return pd.DataFrame(all_scores)


def sample_diverse_examples(score_df, total_samples=100):
    """
    The sampling process
    For each category c_i, sample the top-k safe data (with highest scores)
    k is calculated by total_samples/categories_number
    """
    num_categories = len(score_df.columns)
    per_category = max(1, total_samples // num_categories)

    sampled_indices = set()
    final_samples = []

    for cat in score_df.columns:
        top_indices = score_df[cat].sort_values(ascending=False).index.tolist()
        selected = 0
        for idx in top_indices:
            if idx not in sampled_indices:
                sampled_indices.add(idx)
                final_samples.append((idx, cat, score_df.loc[idx, cat]))
                selected += 1
                if selected >= per_category:
                    break

    if len(final_samples) < total_samples:
        flat_scores = score_df.max(axis=1)
        sorted_indices = flat_scores.sort_values(ascending=False).index.tolist()
        for idx in sorted_indices:
            if idx not in sampled_indices:
                sampled_indices.add(idx)
                top_cat = score_df.loc[idx].idxmax()
                top_score = score_df.loc[idx, top_cat]
                final_samples.append((idx, top_cat, top_score))
                if len(final_samples) >= total_samples:
                    break

    return final_samples


def main():
    args = parse_args()
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Load BeaverTails
    beaver = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    category_vectors, beaver_df = prepare_beaver_category_vectors(beaver, model)

    # Load and embed target dataset
    target = load_dataset("json", data_files="", split="train")
    data_df = embed_target_dataset(target, model, instr_col='instruction', out_col='output')
    
    # Compute cosmean from merged sample
    print("Computing global cosmean from merged random samples...")
    cosmean = compute_cosmean_from_two_sources(
        beaver_df['embedding'].tolist(),
        data_df['embedding'].tolist(),
        sample_size=500
    )
    print(f"cosmean = {cosmean:.4f}")

    # Compute category scores with fixed cosmean
    score_df = compute_point_category_scores(
        data_df, category_vectors,
        normalize=args.normalize,
        cosmean=cosmean
    )
    data_df = data_df.join(score_df)

    # Sample diverse examples across categories
    samples = sample_diverse_examples(score_df, total_samples=args.num_samples)
    subset = []
    category_counts = Counter()

    for idx, cat, score in samples:
        row = data_df.iloc[idx]
        subset.append({
            "instruction": row.get("instruction", row.get("prompt")),
            "output": row.get("output", row.get("response")),
            "input": row.get("input", ""),
            "score": float(score),
            "category": cat
        })
        category_counts[cat] += 1

    with open(args.out, "w") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
    print("Sampled subset saved.")
    print("\nCategory sample counts:")
    for cat, count in category_counts.items():
        print(f"{cat}: {count}")

    print("Sampled subset saved.")



if __name__ == "__main__":
    main()
