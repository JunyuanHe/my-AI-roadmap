# generate.py

import torch
from model import MiniGPT
from config import GPTConfig
from tokenizer import Tokenizer

tokenizer = Tokenizer("AI-generation/my-mini-gpt/mini-gpt-3-RoPE/tokenizer")

# Load the trained model
config = GPTConfig(vocab_size=tokenizer.vocab_size)  # Adjust vocab size if needed
device = config.device
model = MiniGPT(config).to(device)
checkpoint = torch.load("AI-generation/my-mini-gpt/mini-gpt-3-RoPE/model/mini_gpt_3_RoPE.pth")
model.load_state_dict(checkpoint["model_state_dict"])

# Load the tokenizer
tokenizer = Tokenizer("AI-generation/my-mini-gpt/mini-gpt-3-RoPE/tokenizer")

# Generate text
context_text = "Once upon a time"


context_ids = torch.tensor(tokenizer.encode(context_text), dtype=torch.long)[None, :].to(config.device)
generated_ids = model.generate(context_ids, max_new_tokens=400, temperature=0.5)

generated_text = tokenizer.decode(generated_ids[0].tolist())
print(f"Generated Text: {generated_text}")
