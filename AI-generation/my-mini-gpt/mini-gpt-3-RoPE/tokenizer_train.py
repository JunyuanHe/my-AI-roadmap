# tokenizer_train.py

import os
import json
from collections import Counter
from datasets import load_dataset

def train_tokenizer(dataset: str, output_dir: str, vocab_size: int = 1000):
    # Load dataset
    ds = load_dataset(dataset, split="train[:10%]")
    text = "\n".join(ds["text"])

    # Count character frequencies
    char_counts = Counter(text)
    
    # Sort characters by frequency and select the top vocab_size characters
    vocab = ['<UNK>'] + sorted(char_counts, key=char_counts.get, reverse=True)[:vocab_size-1]

    # Create mappings (character -> index and index -> character)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    # Save the tokenizer files
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "stoi.json"), "w") as f:
        json.dump(stoi, f)
    with open(os.path.join(output_dir, "itos.json"), "w") as f:
        json.dump(itos, f)
    
    print(f"Tokenizer saved to {output_dir}")
    return stoi, itos

if __name__ == "__main__":
    # Train and save the tokenizer
    train_tokenizer("roneneldan/TinyStories", output_dir="AI-generation/my-mini-gpt/mini-gpt-3-RoPE/tokenizer")
