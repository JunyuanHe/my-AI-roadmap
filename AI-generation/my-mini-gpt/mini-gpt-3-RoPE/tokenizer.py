# tokenizer.py

import json
import os

# Load the tokenizer
class Tokenizer:
    def __init__(self, tokenizer_dir: str):
        self.stoi = self.load_json(os.path.join(tokenizer_dir, "stoi.json"))
        itos = self.load_json(os.path.join(tokenizer_dir, "itos.json"))
        self.itos = {int(k):v for k,v in itos.items()}
        self.vocab_size = len(self.stoi)

    def load_json(self, path: str):
        with open(path, "r") as f:
            return json.load(f)

    def encode(self, text: str) -> list:
        """Converts text to a list of token indices."""
        return [self.stoi.get(ch, self.stoi["<UNK>"]) for ch in text]  # "<UNK>" for unknown chars

    def decode(self, tokens: list) -> str:
        """Converts token indices back to text."""
        return ''.join([self.itos.get(token, "<UNK>") for token in tokens])  # "<UNK>" for unknown tokens


# Example of how to use this class:
if __name__ == "__main__":
    tokenizer = Tokenizer("tokenizer")  # Load the trained tokenizer from disk
    text = "Once upon a time"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
