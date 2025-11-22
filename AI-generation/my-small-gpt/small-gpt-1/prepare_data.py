import os
import numpy as np
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------
# 配置
# -----------------------------
tokenizer_path = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer.json"
output_dir = "AI-generation/my-small-gpt/small-gpt-1/data"
train_ratio = 0.9  # 90% train, 10% val

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 加载 tokenizer
# -----------------------------
tokenizer = Tokenizer.from_file(tokenizer_path)

bos = tokenizer.token_to_id("<bos>")
eos = tokenizer.token_to_id("<eos>")

assert bos is not None and eos is not None, "Tokenizer 缺少 <bos> 或 <eos>，请检查训练。"

# -----------------------------
# 加载数据集
# -----------------------------
ds = load_dataset("roneneldan/TinyStories", split="train[:10%]")
texts = ds["text"]

print(f"Loaded {len(texts)} texts")


# -----------------------------
# 编码函数（加入 bos/eos）
# -----------------------------
def encode_text(text):
    """把一句文本编码为 token id 列表：<bos> ... <eos>"""
    ids = tokenizer.encode(text).ids
    return [bos] + ids + [eos]


# -----------------------------
# 构造 token 流（train/val）
# -----------------------------
# encoded_texts = [encode_text(t) for t in texts]
encoded_texts = []
for t in tqdm(texts, desc="Encoding", ncols=100):
    encoded_texts.append(encode_text(t))

# 打乱数据
rng = np.random.default_rng(42)
rng.shuffle(encoded_texts)

# 划分训练 / 验证
split_idx = int(len(encoded_texts) * train_ratio)
train_ids = encoded_texts[:split_idx]
val_ids = encoded_texts[split_idx:]

# def flatten(list_of_lists):
#     """把 [[1,2,3], [4,5,6]] -> [1,2,3,4,5,6]"""
#     return [id for seq in list_of_lists for id in seq]

def flatten(list_of_lists):
    for seq in list_of_lists:
        for token in seq:
            yield token

# train_ids = flatten(train_ids)
# val_ids = flatten(val_ids)

print("Flatten train tokens...")
train_ids = list(tqdm(flatten(train_ids), total=sum(len(s) for s in train_ids), ncols=100))

print("Flatten val tokens...")
val_ids = list(tqdm(flatten(val_ids), total=sum(len(s) for s in val_ids), ncols=100))


print(f"Train tokens: {len(train_ids)}")
print(f"Val tokens: {len(val_ids)}")

# -----------------------------
# 保存为 .bin 文件（int16）
# -----------------------------
np.array(train_ids, dtype=np.uint16).tofile(os.path.join(output_dir, "train.bin"))
np.array(val_ids, dtype=np.uint16).tofile(os.path.join(output_dir, "val.bin"))

# 保存 metadata 用于模型加载
meta = {
    "vocab_size": tokenizer.get_vocab_size(),
    "bos_id": bos,
    "eos_id": eos,
}

import json
with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Data preparation finished.")
print("Saved train.bin / val.bin / meta.json.")
