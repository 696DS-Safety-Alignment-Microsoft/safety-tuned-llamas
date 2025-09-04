import argparse
import json
import os
import random
from collections import defaultdict
from datasets import load_dataset
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Sample safe data based on harm categories.")
    parser.add_argument("--data_file", type=str, required=True, help="Input path")
    parser.add_argument("--out", type=str, required=True, help="The output folder")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of total samples to select")
    return parser.parse_args()

def load_data(data_file):
    return load_dataset("json", data_files=data_file, split="train")

def map_categories_to_ids(dataset):
    """
    To efficient store which samples belong to which category
    """
    category_to_ids = defaultdict(list)
    for idx, example in enumerate(dataset):
        for cat, val in example["category"].items():
            if val:
                category_to_ids[cat].append(idx)
    return category_to_ids

def sample_data(dataset, category_to_ids, num_samples, seed):
    # Sort the order of sampling from categories
    # That is, the category with the least samples will be sampled from first, then move on ascendingly
    categories_sorted = sorted(category_to_ids.keys(), key=lambda cat: len(category_to_ids[cat])) 
    num_categories = len(categories_sorted) 
    k = num_samples // num_categories # The number of samples we need for each category

    random.seed(seed) # Create a different random seed for each process
    sampled_ids = set() # Use this to not sample duplicate values
    selected_samples = [] 
    category_counter = Counter() # Keep track how many we sample for each

    for cat in categories_sorted: 
        # Get the samples in cat that are NOT already sampled for other categories
        available_ids = [i for i in category_to_ids[cat] if i not in sampled_ids]
        if len(available_ids) <= k: # If the total sample size is less than k, get all
            chosen = available_ids
        else:
            chosen = random.sample(available_ids, k) # Randomly select k samples from the category
        sampled_ids.update(chosen) 
        category_counter[cat] += len(chosen)

    # After sampling, check if we need to sample more to get to n
    remaining = num_samples - len(sampled_ids)  
    if remaining > 0:
        # Randomly sample for the rest, do not care about category now
        all_available = [i for i in range(len(dataset)) if i not in sampled_ids]
        extra = random.sample(all_available, min(remaining, len(all_available)))
        sampled_ids.update(extra)

    for idx in sampled_ids:
        selected_samples.append(dataset[idx])
        
    # Print out the number sampled for each category and the remaining value
    # This is for the case k > num of samp in category
    print("Category sample counts:") 
    for cat, count in category_counter.items():
        print(f"  {cat}: {count}")
    print(f"Remaining samples filled randomly: {remaining}\n")

    return selected_samples

def save_samples(samples, output_dir, run_idx, num_samples):
    out_path = os.path.join(output_dir, f"sampled{run_idx}_{num_samples}.json")
    with open(out_path, "w") as f:
        json.dump([dict(sample) for sample in samples], f, indent=2)
    print(f"Saved {len(samples)} samples to {out_path}")
    
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    num_samples = args.num_samples
    data_file = 'category_wildguard.json'
    dataset = load_data(args.data_file)
    category_to_ids = map_categories_to_ids(dataset)

    for run in range(10): # 10 run to ensure the randomness
        print(f"\nSampling run {run}:")
        samples = sample_data(dataset, category_to_ids, args.num_samples, seed=run)
        save_samples(samples, args.out, run, args.num_samples)

if __name__ == "__main__":
    main()
