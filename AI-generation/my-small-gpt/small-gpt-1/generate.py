# generate.py

import torch
from model import MiniGPT
from config import GPTConfig
from tokenizer import MyTokenizer

# Load the trained tokenizer
tokenizer_pth = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer.json"
tokenizer = MyTokenizer(tokenizer_pth)  # Make sure this points to the directory with tokenizer files

# Load the trained model
config = GPTConfig(vocab_size=tokenizer.vocab_size)  # Adjust vocab size if needed
device = config.device
model = MiniGPT(config).to(device)
checkpoint = torch.load("AI-generation/my-small-gpt/small-gpt-1/model/mini_gpt_3_RoPE.pth")
model.load_state_dict(checkpoint["model_state_dict"])


# Generate text
context_text = "<bos> Once upon a time"


context_ids = torch.tensor(tokenizer.encode(context_text), dtype=torch.long)[None, :].to(config.device)
generated_ids = model.generate(context_ids, max_new_tokens=400, eos_id=tokenizer.tokenizer.token_to_id("<eos>"), temperature=0.6, top_k=6)

generated_text = tokenizer.decode(generated_ids[0].tolist())
print(f"Generated Text: {generated_text}")
