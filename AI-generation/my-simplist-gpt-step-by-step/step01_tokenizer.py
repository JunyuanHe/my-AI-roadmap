from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder
from tokenizers.pre_tokenizers import Split

# --------------------
# Text Data
# --------------------
texts = [
    "hello world",
    "attention is all you need",
    "transformers are powerful"
]

PAT = r"\w+|[^\w\s]+"


# --------------------
# Build Tokenizer
# --------------------
def build_tokenizer(texts, vocab_size=100):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(Regex(PAT), behavior="isolated")
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "<|endoftext|>"]
    )
    tokenizer.decoder = BPEDecoder()
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer

tokenizer = build_tokenizer(texts)


# --------------------
# Test
# --------------------
s = "transformers are fun!"
enc = tokenizer.encode(s)

print("Text:", s)
print("Token ids:", enc.ids)
print("Decoded:", tokenizer.decode(enc.ids))
print("Vocab size:", tokenizer.get_vocab_size())