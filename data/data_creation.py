import json
import random
import pandas as pd

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
        
def subsample_and_augment(safety_file, alpaca_file, output_prefix, sample_size=100, num_datasets=10, flag="random"):
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


def augment_data(base_data_file, safety_file, output_file, sample_size=100, flag="high"):
    with open(safety_file, 'r') as f:
        safety_data = json.load(f)
    with open(base_data_file, 'r') as f:
        base_data = json.load(f)
    
    safety_data = [
        {
            'instruction': entry.get('instruction', ''),  # Use .get() to avoid KeyErrors
            'output': entry.get('output', ''),
            'input': entry.get('input', '')  # Ensure 'input' exists; otherwise, default to ''
        }
        for entry in safety_data
    ]
    for item in safety_data:
        item["safety_flag"] = True
    
    for item in base_data:
        item["safety_flag"] = False
        
    if flag == "high":
        safety_data = safety_data[:sample_size]
    elif flag == "low":
        safety_data = safety_data[-sample_size:]
    else:
        safety_data = random.sample(safety_data, sample_size)
    
    augmented_data = base_data + safety_data
    random.shuffle(augmented_data)
    
    with open(output_file, 'w') as f:
        json.dump(augmented_data, f, indent=4)  
    
    print(f"Saved: {output_file}")
        
def modify_safety_flag(data_file, safety_file):
# Load safety data
    with open(safety_file, 'r') as f:
        safety_data = json.load(f)

    # Print the number of samples in safety data
    print(f"Number of samples in safety file: {len(safety_data)}")

    # Convert safety data to a set for fast lookup
    safety_set = set((item['instruction'], item.get('input', ''), item['output']) for item in safety_data)

    # Load input data
    with open(data_file, 'r') as f:
        data = json.load(f)

    # Count matching samples and add safety flag
    match_count = 0
    for item in data:
        if (item['instruction'], item.get('input', ''), item['output']) in safety_set:
            item['safety_flag'] = True
            match_count += 1
        else:
            item['safety_flag'] = False

    # Print the number of matched samples
    print(f"Number of matching samples: {match_count}")

    # If match_count is correct, dump updated data into JSON
    if match_count == 100:
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Updated data with safety flags saved to {data_file}")
    else:
        print("Match count is incorrect, not saving the file.")
    
# Example usage
#modify_safety_flag('./training/subsample_100/alpaca_safer_100_dataset_4.json', "./training/safety_only_data_Instructions.json")
#add_safety_flag("./training/safety_only_data_Instructions.json")
#subsample_and_augment("./training/safety_only_data_Instructions.json", "./training/alpaca_small.json", "./training/temp_folder", 100)
augment_data("./training/alpaca_small.json", '../../outputs/harm_sorted_safety_data_with_scores.json', "./training/subsample_100/harm_selection_100_low.json", 100, "low")
