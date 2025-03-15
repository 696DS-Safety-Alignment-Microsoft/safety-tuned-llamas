import json
import random

def add_safety_flag(json_file):
    """Modifies a JSON file by adding 'safety': False to each dictionary in the list."""
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Add "safety": False to each dictionary
    for item in data:
        item["safety_flag"] = True

    # Save the modified JSON back to the file
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
        
def subsample_and_augment(safety_file, alpaca_file, output_prefix, sample_size=100, num_datasets=10):
    # Load safety-only data
    with open(safety_file, 'r') as f:
        safety_data = json.load(f)
    
    # Load alpaca data
    with open(alpaca_file, 'r') as f:
        alpaca_data = json.load(f)
    
    for i in range(num_datasets):
        sampled_data = random.sample(safety_data, sample_size)
        augmented_data = alpaca_data + sampled_data
        random.shuffle(augmented_data)
    
        output_file = f"{output_prefix}/alpaca_safer_{sample_size}_dataset_{i}.json"
        with open(output_file, 'w') as f:
            json.dump(augmented_data, f, indent=4)
        
        print(f"Saved: {output_file}")

    
# Example usage
#add_safety_flag("./training/safety_only_data_Instructions.json")
subsample_and_augment("./training/safety_only_data_Instructions.json", "./training/alpaca_small.json", "./training/temp_folder", 100)
