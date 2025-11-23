import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm



# Globals for workers (will be set in initializer)
TOK = None
BOS = None
EOS = None

# ----------------------------------
# Worker initializer: runs once per worker process
# ----------------------------------
def worker_init(tokenizer_path, bos_id, eos_id):
    global TOK, BOS, EOS
    TOK = Tokenizer.from_file(tokenizer_path)
    BOS = bos_id
    EOS = eos_id

# Re-define the worker encoder here (so it uses TOK set by worker_init)
def encode_batch_worker(text_batch):
    encs = TOK.encode_batch(text_batch)
    return [[BOS] + e.ids + [EOS] for e in encs]


if __name__ == '__main__':
    # -----------------------------
    # 配置
    # -----------------------------
    tokenizer_path = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer.json"
    output_dir = "AI-generation/my-small-gpt/small-gpt-1/data"
    train_ratio = 0.9  # 90% train, 10% val

    os.makedirs(output_dir, exist_ok=True)


    # Multiprocessing settings
    batch_size = 2048      # batch_size larger is faster（2k ~ 10k 都可以）
    num_proc = 8 # use all CPU

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
    ds = load_dataset("roneneldan/TinyStories", split="train[:100%]+validation[:100%]")
    texts = ds["text"]
    total = len(texts)
    print(f"Loaded {total} texts")

    # ----------------------------------
    # Prepare batched inputs
    # ----------------------------------
    batched_texts = [texts[i : i + batch_size] for i in range(0, total, batch_size)]
    print(f"Encoding in batches (batch_size={batch_size}, processes={num_proc})...")

    # ----------------------------------
    # 分批处理，并行加速构造 token 流
    # ----------------------------------
    encoded_texts = []

    with Pool(
        processes=num_proc,
        initializer=worker_init,
        initargs=(tokenizer_path, bos, eos)
    ) as pool:
        for token_seqs in tqdm(
            pool.imap_unordered(encode_batch_worker, batched_texts),
            total=len(batched_texts),
            ncols=100,
            desc="Encoding"
        ):
            encoded_texts.extend(token_seqs)

    # ----------------------------------
    # 打乱数据
    # ----------------------------------
    rng = np.random.default_rng(42)
    rng.shuffle(encoded_texts)

    # ----------------------------------
    # 划分 train / val
    # ----------------------------------
    split_idx = int(len(encoded_texts) * train_ratio)
    train_ids = encoded_texts[:split_idx]
    val_ids = encoded_texts[split_idx:]


    def flatten_and_save(seqs, out_path, desc="Flattening"):
        """使用 memmap 边写边 flatten，并显示 tqdm 进度条。"""
        total_len = sum(len(s) for s in seqs)

        # 创建 memmap（二进制文件）
        arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_len,))

        idx = 0
        for s in tqdm(seqs, total=len(seqs), ncols=100, desc=desc):
            arr[idx:idx+len(s)] = s
            idx += len(s)

        # 刷盘
        arr.flush()
        del arr  # 关闭 memmap
        return total_len

    # Flatten train / val and save
    train_bin = os.path.join(output_dir, "train.bin")
    val_bin  = os.path.join(output_dir, "val.bin")

    train_token_len = flatten_and_save(train_ids, train_bin, desc="Flatten train")
    val_token_len = flatten_and_save(val_ids, val_bin, desc="Flatten val")

    print("Binary files written.")


    print(f"Train tokens: {train_token_len}")
    print(f"Val tokens: {val_token_len}")


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
