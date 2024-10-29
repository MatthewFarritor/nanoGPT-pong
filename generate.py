import os
import torch
import pickle
import numpy as np

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Adjust these paths as needed
out_dir = 'out-my_dataset'  # Output directory where the model checkpoint is saved
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
data_dir = 'data/my_dataset'  # Directory where your dataset and meta.pkl are located

device = 'cpu'  # Use 'cpu' or 'cuda' depending on your setup

# -----------------------------------------------------------------------------
# Load the Model and Meta Information
# -----------------------------------------------------------------------------

# Load the metadata (e.g., stoi and itos dictionaries)
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']
vocab_size = meta['vocab_size']

# Load the trained model checkpoint
checkpoint = torch.load(ckpt_path, map_location=device)

# Initialize the model configuration
# Only include the keys that GPTConfig accepts
model_args = checkpoint['config']
model_args['vocab_size'] = vocab_size  # Ensure vocab_size matches your data

# Define the keys that GPTConfig accepts
gptconfig_args = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout']

# Filter model_args to include only valid keys
filtered_model_args = {k: model_args[k] for k in gptconfig_args if k in model_args}

# Initialize the GPT configuration
gptconf = GPTConfig(**filtered_model_args)

# Initialize the model
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()  # Set model to evaluation mode

# -----------------------------------------------------------------------------
# Text Generation Function
# -----------------------------------------------------------------------------

def generate(start_sequence='', max_new_tokens=100, temperature=1.0, top_k=None):
    # Encode the starting sequence
    if start_sequence:
        start_ids = [stoi.get(ch, 0) for ch in start_sequence]
    else:
        start_ids = [np.random.randint(vocab_size)]  # Start with a random token if no start_sequence
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, sequence_length)
    
    # Generate tokens
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Ensure x is at most block_size in length
            x_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
            logits, _ = model(x_cond)
            # Focus on the last time step
            logits = logits[:, -1, :] / temperature  # (1, vocab_size)
            # Optionally apply top-k filtering
            if top_k is not None:
                v, ix = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Sample from the distribution
            next_id = torch.multinomial(probs, num_samples=1).item()
            # Append to the sequence
            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
    
    # Decode the generated sequence
    generated_ids = x[0].tolist()
    generated_text = ''.join([itos.get(i, '') for i in generated_ids])
    return generated_text

# -----------------------------------------------------------------------------
# Generate and Save Output
# -----------------------------------------------------------------------------

# Define a starting sequence or leave it empty
start_sequence = ''  # You can provide a starting sequence relevant to your data

# Generate text with 100 tokens
generated_text = generate(
    start_sequence=start_sequence,
    max_new_tokens=10000,  # Generate 100 tokens
    temperature=1.0,     # Adjust temperature for randomness
    top_k=5              # Adjust top_k for controlling diversity
)

# Save the generated text to a file
output_file = os.path.join(out_dir, 'generated_output.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(generated_text)

print(f"Generated text saved to {output_file}")
print("Generated Text:")
print(generated_text)
