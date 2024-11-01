import os
import numpy as np

out_dir = 'out-my_dataset'

eval_interval = 500
eval_iters = 200
log_interval = 10

learning_rate = 1e-3
batch_size = 64
block_size = 20  

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1
bias = True

dataset = 'my_dataset'
gradient_accumulation_steps = 1

device = 'cpu'     
dtype = 'float32'   


compile = False 

data_dir = os.path.join('data', dataset)
train_data_path = os.path.join(data_dir, 'train.bin')
train_data = np.fromfile(train_data_path, dtype=np.uint16)

num_epochs = 1 

max_iters = (len(train_data) // (batch_size * block_size)) * num_epochs

if max_iters == 0:
    max_iters = 1

print(f"Calculated max_iters: {max_iters}")
