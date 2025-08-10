import json
import random
import numpy as np
from pathlib import Path

# ---------------------------
# User settings
# ---------------------------

# Path to your original dataset (JSON format)
input_path = Path("fitbit_dataset_expanded_20x.json")

# Output path for the expanded dataset
output_path = Path("fitbit_dataset_expanded_80x.json")

# Noise standard deviation for synthetic data (small perturbations)
noise_scale = 0.02

# ---------------------------
# Load original dataset
# ---------------------------
with open(input_path, "r") as f:
    original_data = json.load(f)

# Convert to NumPy array for vectorized operations
original_array = np.array(original_data)
original_len = len(original_array)

# ---------------------------
# Determine target size
# ---------------------------
desired_len = original_len * 4
synthetic_needed = desired_len - original_len

# ---------------------------
# Generate synthetic samples
# ---------------------------
synthetic_data = []
for _ in range(synthetic_needed):
    # Pick a random row from the dataset
    base_sample = random.choice(original_array)

    # Add small multiplicative noise
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=base_sample.shape)
    synthetic_sample = base_sample * (1 + noise)

    # Store as a list for JSON serialization
    synthetic_data.append(synthetic_sample.tolist())

# ---------------------------
# Combine and save
# ---------------------------
expanded_dataset = original_data + synthetic_data

with open(output_path, "w") as f:
    json.dump(expanded_dataset, f)

print(f"Expanded dataset saved to: {output_path}")
print(f"Original size: {original_len}  |  New size: {len(expanded_dataset)}")

