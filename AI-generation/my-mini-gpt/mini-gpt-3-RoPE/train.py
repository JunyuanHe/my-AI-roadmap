# train.py

import torch
import os
from model import MiniGPT
from config import GPTConfig
from tokenizer import Tokenizer
from datasets import load_dataset
from utils import get_batch
from torchsummary import summary


# Tokenize

# Load the trained tokenizer
tokenizer = Tokenizer("AI-generation/my-mini-gpt/mini-gpt-3-RoPE/tokenizer")  # Make sure this points to the directory with tokenizer files

# Load dataset and tokenize
ds = load_dataset("roneneldan/TinyStories", split="train[:100%]")
text = "\n".join(ds["text"])

# Convert the text to token IDs using the tokenizer
token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Setup model and training loop as before
config = GPTConfig(vocab_size=tokenizer.vocab_size)
device = config.device
model = MiniGPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

steps = 5000
batch_size = 64
seq_len = config.seq_len

summary(model, input_size=(seq_len,))  # Only specify (seq_len,) here

# Training loop
for step in range(steps):
    xb, yb = get_batch(token_ids, batch_size, seq_len, config)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

# Save the model after training
torch.save({
    'epoch': steps,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'AI-generation/my-mini-gpt/mini-gpt-3-RoPE/model/mini_gpt_3_RoPE.pth')
