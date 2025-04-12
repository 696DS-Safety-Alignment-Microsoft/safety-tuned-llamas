import json
import csv
import os
import argparse
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def main():
    parser = argparse.ArgumentParser(description="Compute harmfulness scores from JSON and save to CSV.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output CSV")

    args = parser.parse_args()
    
    # Load JSON file
    with open(args.input_json, "r") as file:
        data = json.load(file)
    
    #print("loaded json")
    # Extract instructions and outputs
    user_texts = data["instructions"]
    assistant_texts = data["outputs"]
    
    # Compute helpfulness scores
    reward_name = "OpenAssistant/reward-model-deberta-v3-large"
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
    
    reward_scores = []
    for instruction, output in zip(user_texts, assistant_texts):
        inputs = tokenizer(instruction, output, return_tensors='pt')
        with torch.no_grad():
            score = rank_model(**inputs).logits[0].cpu().item()
        reward_scores.append(score)
    
    #print("computed scores", reward_scores)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    
    # Write to CSV file
    with open(args.output_dir, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Prompt", "Response", "Helpfulness Score"])  # Header
        for instr, output, score in zip(user_texts, assistant_texts, reward_scores):
            writer.writerow([instr,output,score])

    print(f"CSV saved at: {args.output_dir}")
    print(f"Average helpfulness score: {np.mean(reward_scores):.4f}")

if __name__ == "__main__":
    main()