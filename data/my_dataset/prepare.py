# data/my_dataset/prepare.py

import os
import pickle
import numpy as np

# Path to your data directory
data_dir = os.path.dirname(__file__)
input_file = os.path.join(data_dir, 'input.txt')

# Read the data
with open(input_file, 'r', encoding='utf-8') as f:
    data = f.read()

# Print a sample of the data
print("Sample data:")
print(data[:500])  # Adjust the number as needed

print(f"Length of dataset in characters: {len(data):,}")

# Get all unique characters in the data
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"All unique characters: {''.join(chars)}")
print(f"Vocabulary size: {vocab_size}")

# Create mappings from characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Encode the entire dataset and store it as a numpy array
data_ids = np.array([stoi[ch] for ch in data], dtype=np.uint16)
print(f"Data has {len(data_ids):,} tokens.")

# Split the data into train and validation sets
n = int(0.9 * len(data_ids))
train_ids = data_ids[:n]
val_ids = data_ids[n:]

# Save the data to bin files
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

# Save the metadata to a pickle file
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete.")
