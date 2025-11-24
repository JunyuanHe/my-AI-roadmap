import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders, normalizers

# -----------------------------
# 配置
# -----------------------------
train_file = "AI-generation/my-small-gpt/data/simplebooks/simplebooks-92-raw/train.txt"
tokenizer_pth = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer-simplebooks.json"
vocab_size = 2000  # 根据需求调整
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
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()  # add_prefix_space=False
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# str = tokenizer.normalizer.normalize_str("When the things were all taken out of the ship, Captain Solomon had his rowboat let down into the water")

# str = tokenizer.pre_tokenizer.pre_tokenize_str("When the things were all taken out of the ship, Captain Solomon had his rowboat let down into the water")
print(str)

# 使用 <pad>, <unk>, <eob> 作为 special tokens
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=2,
    # initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<pad>", "<unk>", "<eob>"]
)

# -----------------------------
# 训练 tokenizer
# -----------------------------
tokenizer.train_from_iterator(batch_iterator(train_file, batch_size=batch_size), trainer=trainer)

# 测试 tokenizer
sentence = "Let's test\nthis tokenizer."
encoding = tokenizer.encode(sentence)
decoding = tokenizer.decode(encoding.ids)
print(encoding)
print(decoding)

# -----------------------------
# 保存 tokenizer
# -----------------------------
os.makedirs(os.path.dirname(tokenizer_pth), exist_ok=True)
tokenizer.save(tokenizer_pth)

print(f"Tokenizer saved to {tokenizer_pth}")
