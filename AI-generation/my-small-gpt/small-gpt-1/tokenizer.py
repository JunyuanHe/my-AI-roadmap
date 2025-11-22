# tokenizer.py

import json
import os
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Load the tokenizer
class MyTokenizer:
    def __init__(self, tokenizer_dir: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_dir)
        self.tokenizer.decoder = ByteLevelDecoder(add_prefix_space=False)  # 去除Byte-Level pretokenizer导致的前导空格
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list:
        """Converts text to a list of token indices."""
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: list) -> str:
        """Converts token indices back to text."""
        return self.tokenizer.decode(tokens, skip_special_tokens=False)  # , skip_special_tokens=True


# Example of how to use this class:
if __name__ == "__main__":
    tokenizer = MyTokenizer("AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer.json")  # Load the trained tokenizer from disk
    text = "<bos> There was a girl"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
