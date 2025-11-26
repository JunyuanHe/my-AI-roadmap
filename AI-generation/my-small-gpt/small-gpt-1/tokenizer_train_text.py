import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders, normalizers, Regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# PAT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# -----------------------------
# 配置
# -----------------------------
# train_file = "AI-generation/my-small-gpt/data/simplebooks/simplebooks-92-raw/train.txt"
# tokenizer_pth = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer-simplebooks.json"
train_file = "AI-generation/my-small-gpt/data/tiny-stories-v2/TinyStoriesV2-GPT4-train.txt"
tokenizer_pth = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer-tinystoriesv2.json"
vocab_size = 5000  # 根据需求调整
batch_size = 1000

# -----------------------------
# 读取训练文本
# -----------------------------
def batch_iterator(file_path, batch_size=batch_size):
    """
    按行或按 batch_size 切分大文本，返回每个 batch 的 list[str]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        buffer = []
        for line in f:
            # line = line.strip()
            if not line:
                continue
            buffer.append(line)
            if len(buffer) >= batch_size:
                yield buffer
                buffer = []
        if buffer:
            yield buffer

# -----------------------------
# 创建 tokenizer
# -----------------------------
# tokenizer = Tokenizer(models.BPE())
tokenizer = Tokenizer(models.BPE(byte_fallback=True))
# Byte-level pretokenizer
# tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)  # add_prefix_space=False

# gpt pretokenizer
split_pre_tokenizer = pre_tokenizers.Split(
    Regex(PAT), behavior="isolated", invert=False
)
byte_pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False, use_regex=False
)
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [split_pre_tokenizer, byte_pre_tokenizer]
)
# tokenizer.pre_tokenizer = split_pre_tokenizer

# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# tokenizer.decoder = decoders.ByteLevel()
tokenizer.decoder = decoders.Sequence(
    [decoders.ByteLevel(), decoders.ByteFallback()]
)
# tokenizer.decoder = decoders.ByteFallback()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# str = tokenizer.normalizer.normalize_str("When the things were all taken out of the ship, Captain Solomon had his rowboat let down into the water")

str = tokenizer.pre_tokenizer.pre_tokenize_str("When the things were all taken out of the ship, Captain Solomon had his rowboat let down into the water")
print(str)

# 使用 <pad>, <unk>, <eob> 作为 special tokens
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    show_progress=True,
    min_frequency=2,
    # initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    # special_tokens=["<pad>", "<unk>", "<eob>"] + [f"<0x{i:02X}>" for i in range(256)],  # seed sm vocab
    special_tokens=["<|endoftext|>"] + [f"<0x{i:02X}>" for i in range(256)],  # seed sm vocab
)

# -----------------------------
# 训练 tokenizer
# -----------------------------
tokenizer.train_from_iterator(batch_iterator(train_file, batch_size=batch_size), trainer=trainer)

# 测试 tokenizer
sentence = "Let's test\nthis tokenizer, café."
encoding = tokenizer.encode(sentence)
decoding = tokenizer.decode(encoding.ids)
print(encoding)
print(decoding)

print(tokenizer.encode("A").tokens)
print(tokenizer.encode("é").tokens)

# -----------------------------
# 保存 tokenizer
# -----------------------------
os.makedirs(os.path.dirname(tokenizer_pth), exist_ok=True)
tokenizer.save(tokenizer_pth)

print(f"Tokenizer saved to {tokenizer_pth}")
