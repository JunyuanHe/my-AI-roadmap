# train.py

import torch
import os
import math
from torch.optim.lr_scheduler import LambdaLR
from model import MiniGPT
from config import GPTConfig
from tokenizer import MyTokenizer
from datasets import load_dataset
from utils import get_batch
from torchsummary import summary
import numpy as np


# Tokenize

# Load the trained tokenizer
tokenizer_pth = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer.json"
tokenizer = MyTokenizer(tokenizer_pth)  # Make sure this points to the directory with tokenizer files

# Load dataset
# ds = load_dataset("roneneldan/TinyStories", split="train[:100%]")
# text = "\n".join(ds["text"])
token_ids = np.memmap('AI-generation/my-small-gpt/small-gpt-1/data/train.bin', dtype=np.uint16, mode='r')


# Convert the text to token IDs using the tokenizer
# token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Setup model and training loop as before
config = GPTConfig(vocab_size=tokenizer.vocab_size)
device = config.device



def lr_lambda(step):
    if step < config.warmup_steps:
        return step / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))



model = MiniGPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, betas=(0.9, 0.95))

# scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=1000,       # number of steps in the first restart
    T_mult=2,      # multiply T_i by this after each restart
    eta_min=config.min_lr # minimum learning rate
)

# scheduler = LambdaLR(optimizer, lr_lambda)


steps = config.max_steps
batch_size = 64
seq_len = config.seq_len

summary(model, input_size=(seq_len,))  # Only specify (seq_len,) here

# Training loop
for step in range(steps):
    xb, yb = get_batch(token_ids, batch_size, seq_len, config)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    # Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    # lr schedule: warmup + cosine restart
    if step < config.warmup_steps:
        lr = config.max_lr * step / config.warmup_steps
        for g in optimizer.param_groups:
            g["lr"] = lr
    else:
        scheduler.step()

    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

# Save the model after training
# Ensure the directory exists
save_path = 'AI-generation/my-small-gpt/small-gpt-1/model/small_gpt_1.pth'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save({
    'epoch': steps,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, save_path)
