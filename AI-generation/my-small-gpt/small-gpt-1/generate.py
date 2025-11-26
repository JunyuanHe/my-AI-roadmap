# generate.py

import os
import torch
from model import MiniGPT
from config import GPTConfig
from tokenizer import MyTokenizer


# Load the trained tokenizer
tokenizer_pth = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer-tinystoriesv2.json"
# tokenizer_pth = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer-simplebooks.json"
# tokenizer_pth = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer.json"
tokenizer = MyTokenizer(tokenizer_pth)  # Make sure this points to the directory with tokenizer files

# Load the trained model
config = GPTConfig(vocab_size=tokenizer.vocab_size)
device = config.device
model = MiniGPT(config).to(device)
model_path = os.path.join(config.model_dir, "small_gpt_1_best.pth")
checkpoint = torch.load(model_path)
# print(checkpoint["model_state_dict"].keys())

model.load_state_dict(checkpoint["model_state_dict"])

# eos = "<eob>"
eos = "<|endoftext|>"


# Generate text
context_text = "The two boys"
print(f"Input Text: {context_text}")
# print(tokenizer.decode(tokenizer.encode(context_text)))

context_ids = torch.tensor(tokenizer.encode(context_text), dtype=torch.long)[None, :].to(config.device)
# print(context_ids)
generated_ids = model.generate(context_ids, max_new_tokens=400, eos_id=tokenizer.tokenizer.token_to_id(eos), temperature=0.6, top_k=6)

# print(generated_ids)




generated_text = tokenizer.decode(generated_ids[0].tolist())
print(f"Generated Text: {generated_text}")
