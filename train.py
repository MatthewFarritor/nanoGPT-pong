# train.py

"""
This training script has been modified to ensure that the model trains on the entire dataset.
The data loader has been changed to iterate over the dataset sequentially, and max_iters is
calculated based on the dataset size to include all the data in the training process.
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default configuration values
# These can be overridden by passing a configuration file
# -----------------------------------------------------------------------------
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

# WandB logging
wandb_log = False  # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2'  # 'run' + str(time.time())

# Data
dataset = 'my_dataset'
gradient_accumulation_steps = 1  # Adjusted to 1
batch_size = 64  # Adjusted as needed
block_size = 2  # Adjusted as needed

# Model
n_layer = 4  # Adjusted as needed
n_head = 4   # Adjusted as needed
n_embd = 128  # Adjusted as needed
dropout = 0.1  # Adjusted as needed
bias = True  # Adjusted as needed

# AdamW optimizer
learning_rate = 1e-3  # Adjusted as needed
max_iters = None  # Will be set after loading the config
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # Clip gradients at this value, or disable if == 0.0

# Learning rate decay settings
decay_lr = False  # Set to False since we have a small dataset
warmup_iters = 0  # No warmup
lr_decay_iters = None  # Will be set to max_iters if decay_lr is True
min_lr = 1e-5  # Minimum learning rate

# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.

# System
device = 'cpu'  # 'cpu' or 'cuda', adjusted in the config file
dtype = 'float32'  # 'float32', 'bfloat16', or 'float16', adjusted in the config file
compile = False  # Adjusted in the config file
# -----------------------------------------------------------------------------

# Override defaults from configuration file
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
if len(sys.argv) > 1:
    exec(open(sys.argv[1]).read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# Setup DDP if needed
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a DDP run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # Adjust gradient accumulation steps for DDP
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # Single process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"Tokens per iteration: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow TF32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'

# Set dtype and context manager
if device_type == 'cpu':
    ptdtype = torch.float32  # Use float32 on CPU
else:
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Initialize GradScaler if using mixed precision
if device_type == 'cuda':
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
else:
    scaler = None

# -----------------------------------------------------------------------------
# Data loading and preparation
# -----------------------------------------------------------------------------

# Load data
data_dir = os.path.join('data', dataset)
train_data_path = os.path.join(data_dir, 'train.bin')
val_data_path = os.path.join(data_dir, 'val.bin')

# Use np.memmap for memory efficiency
train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')

# Initialize data indices
train_idx = 0
val_idx = 0

# Define get_batch function for sequential data loading
def get_batch(split):
    global train_data, val_data, train_idx, val_idx
    data = train_data if split == 'train' else val_data
    idx = train_idx if split == 'train' else val_idx
    # Prepare batches
    x_list = []
    y_list = []
    for _ in range(batch_size):
        if idx >= len(data) - block_size - 1:
            idx = 0  # Reset index at the end of the data
        x = torch.from_numpy(data[idx:idx+block_size].astype(np.int64))
        y = torch.from_numpy(data[idx+1:idx+1+block_size].astype(np.int64))
        x_list.append(x.unsqueeze(0))
        y_list.append(y.unsqueeze(0))
        idx += block_size  # Move index forward
    # Stack batches
    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    # Update global index
    if split == 'train':
        train_idx = idx
    else:
        val_idx = idx
    # Move to device
    x = x.to(device)
    y = y.to(device)
    return x, y

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"Found vocab_size = {vocab_size} (inside {meta_path})")
else:
    raise FileNotFoundError(f"meta.pkl not found in {data_dir}")

# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------

# Model configuration
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=vocab_size,  # Set vocab_size from meta.pkl
    dropout=dropout,
)

# Initialize or load the model
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
else:
    raise NotImplementedError("Only 'scratch' initialization is supported in this script.")

# Move model to device
model.to(device)
print(f"Number of parameters: {model.get_num_params()/1e6:.2f}M")

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

# Determine max_iters if not set
if max_iters is None:
    # Load train data to calculate max_iters
    train_data_len = len(train_data)
    max_iters = (train_data_len // (batch_size * block_size))
    num_epochs = 1  # Set number of epochs
    max_iters = max_iters * num_epochs
    print(f"Calculated max_iters: {max_iters}")
if lr_decay_iters is None:
    lr_decay_iters = max_iters

# Learning rate scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Initialize variables before training loop
iter_num = 0
best_val_loss = float('inf')
raw_model = model.module if ddp else model  # Unwrap DDP
running_mfu = -1.0
t0 = time.time()
local_iter_num = 0  # Initialize local_iter_num

# Helper function for evaluation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# Training loop
while True:
    # Determine learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Get batch
    X, Y = get_batch('train')

    # Forward pass
    with ctx:
        logits, loss = model(X, Y)

    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    # Logging
    if iter_num % log_interval == 0 and master_process:
        loss_value = loss.item()
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if local_iter_num >= 5:  # Let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        else:
            running_mfu = -1.0
        print(f"Iter {iter_num}: loss {loss_value:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

    # Evaluation
    if iter_num % eval_interval == 0 and master_process:
        model.eval()
        losses = estimate_loss()
        model.train()
        print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"Saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    iter_num += 1
    local_iter_num += 1  # Increment local_iter_num

    # Termination condition
    if iter_num >= max_iters:
        print("Training complete.")
        break

# Clean up
if ddp:
    destroy_process_group()
