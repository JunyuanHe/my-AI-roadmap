import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tokenizers import Tokenizer
from tqdm import tqdm

# Globals for workers (will be set in initializer)
TOK = None

# ----------------------------------
# Worker initializer: runs once per worker process
# ----------------------------------
def worker_init(tokenizer_path):
    global TOK
    TOK = Tokenizer.from_file(tokenizer_path)

# Re-define the worker encoder here (so it uses TOK set by worker_init)
def encode_batch_worker(text_batch):
    encs = TOK.encode_batch(text_batch)
    return [e.ids for e in encs]

if __name__ == '__main__':
    # -----------------------------
    # 配置
    # -----------------------------
    # tokenizer_path = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer-simplebooks.json"
    # data_dir = "AI-generation/my-small-gpt/data/simplebooks/simplebooks-92-raw"
    tokenizer_path = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer-tinystoriesv2.json"
    data_dir = "AI-generation/my-small-gpt/data/tiny-stories-v2"
    output_dir = "AI-generation/my-small-gpt/small-gpt-1/data"

    os.makedirs(output_dir, exist_ok=True)

    batch_size = 20480
    num_proc = min(8, cpu_count())

    # -----------------------------
    # 读取文本文件
    # -----------------------------
    def read_text_file(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    # train_text = read_text_file(os.path.join(data_dir, "train.txt"))
    # val_text   = read_text_file(os.path.join(data_dir, "valid.txt"))
    train_text = read_text_file(os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt"))
    val_text   = read_text_file(os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt"))


    print(f"Train text length: {len(train_text)} chars")
    print(f"Val text length: {len(val_text)} chars")

    # -----------------------------
    # 将文本拆成 batch 以便并行 tokenize
    # -----------------------------
    def chunk_text(text, chunk_size):
        # 按字符数切分大文本为 chunk
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Split the texts into batches
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    

    train_chunks = chunk_text(train_text, batch_size*50)  # 估算每 chunk 大小
    val_chunks   = chunk_text(val_text, batch_size*50)

    train_batches = [list(chunks(text, batch_size)) for text in train_chunks]
    val_batches = [list(chunks(text, batch_size)) for text in val_chunks]

    # print(val_chunks[:3])

    # worker_init(tokenizer_path)

    # train_ids = encode_batch_worker(train_chunks)
    # val_ids = encode_batch_worker(val_chunks)

    # print(val_ids[0:2])

    # Save
    # np.array(train_ids, dtype=np.uint16).tofile(os.path.join(output_dir, "train-simplebooks.bin"))
    # np.array(val_ids, dtype=np.uint16).tofile(os.path.join(output_dir, "val-simplebooks.bin"))
    

    # # -----------------------------
    # 并行 tokenize
    # -----------------------------
    def tokenize_chunks(chunks, desc="Tokenizing"):
        token_ids = []
        with Pool(processes=num_proc, initializer=worker_init, initargs=(tokenizer_path,)) as pool:
            for batch in tqdm(pool.imap_unordered(encode_batch_worker, chunks), total=len(chunks), ncols=100, desc=desc):
                token_ids.extend(batch)
        return token_ids

    train_ids = tokenize_chunks(train_batches, desc="Tokenizing train")
    val_ids   = tokenize_chunks(val_batches, desc="Tokenizing val")

    # # -----------------------------
    # # Flatten 并保存到二进制文件
    # # -----------------------------
    def flatten_and_save(seqs, out_path, desc="Flattening"):
        total_len = sum(len(s) for s in seqs)
        arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_len,))
        idx = 0
        for s in tqdm(seqs, total=len(seqs), ncols=100, desc=desc):
            arr[idx:idx+len(s)] = s
            idx += len(s)
        arr.flush()
        del arr
        return total_len

    train_bin = os.path.join(output_dir, "train-simplebooks.bin")
    val_bin   = os.path.join(output_dir, "val-simplebooks.bin")

    train_token_len = flatten_and_save(train_ids, train_bin, desc="Flatten train")
    val_token_len   = flatten_and_save(val_ids, val_bin, desc="Flatten val")

    print(f"Train tokens: {train_token_len}")
    print(f"Val tokens: {val_token_len}")

    # -----------------------------
    # 保存 metadata
    # -----------------------------
    tokenizer = Tokenizer.from_file(tokenizer_path)
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "bos_id": None,
        "eos_id": None,
    }

    import json
    with open(os.path.join(output_dir, "meta-simplebooks.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Data preparation finished.")
    print("Saved train.bin / val.bin / meta.json.")
