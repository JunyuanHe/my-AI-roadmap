import torch
from config import GPTConfig
import numpy as np

def get_batch(data, batch_size, seq_len, config: GPTConfig):
    """Generates a batch of input-output sequences from the data."""
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+seq_len+1].astype(np.int64)) for i in ix])
    return x.to(config.device), y.to(config.device)
