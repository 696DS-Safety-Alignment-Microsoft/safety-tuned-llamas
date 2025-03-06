import json
import csv
import os
import argparse
import numpy as np
from HarmfulnessRewardModel import AbsoluteHarmfulnessPredictor

def main():
    parser = argparse.ArgumentParser(description="Compute harmfulness scores from JSON and save to CSV.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output CSV")

    args = parser.parse_args()
    
    # Load JSON file
    with open(args.input_json, "r") as file:
        data = json.load(file)

    # Extract instructions and outputs
    user_texts = data["instructions"]
    assistant_texts = data["outputs"]

    # Compute harmfulness scores
    setup = "redteam"  # or "redteam-osst"
    harmfulness_predictor = AbsoluteHarmfulnessPredictor(setup, device="cuda:0")
    harmfulness_scores = harmfulness_predictor.predict(user_texts, assistant_texts)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    # Write to CSV file
    with open(args.output_dir, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Harmfulness Score"])  # Header
        for instr, output, score in zip(user_texts, assistant_texts, harmfulness_scores):
            writer.writerow([score])

    print(f"CSV saved at: {args.output_dir}")
    print(f"Average harmfulness score: {np.mean(harmfulness_scores):.4f}")

if __name__ == "__main__":
    main()