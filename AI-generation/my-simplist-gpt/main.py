import torch
import torch.nn as nn

# ==================
#  Tokenizer
# ==================

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
from tokenizers.decoders import BPEDecoder

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def build_tokenizer(texts, vocab_size=2000):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(Regex(PAT), behavior="isolated")
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "<|endoftext|>"]
    )
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.decoder = BPEDecoder()
    return tokenizer

# ==================
#  Dataset
# ==================

from datasets import Dataset
import torch

def tokenize_dataset(dataset, tokenizer, eos="<|endoftext|>"):
    """Concatenate all documents and tokenize into one long list of token ids."""
    all_ids = []
    eos_token_id = tokenizer.token_to_id(eos)
    for text in dataset["text"]:
        ids = tokenizer.encode(text).ids
        ids.append(eos_token_id)  # add eos token
        all_ids.extend(ids)
    return all_ids

# Example tiny dataset
# texts = [
#     "hello world.",
#     "machine learning is fun.",
#     "transformers are powerful.",
#     "attention is all you need.",
# ]

from datatexts import texts

def load_texts_from_file(filepath):
    """
    Reads a .txt file and returns a list of cleaned text lines.
    
    Args:
        filepath (str): Path to the text file.
        
    Returns:
        list[str]: List of text entries.
    """
    texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:               # skip empty lines
                texts.append(line)
    return texts


dataset = Dataset.from_dict({"text": texts})

# Train tokenizer
vocab_size = 500
tokenizer = build_tokenizer(texts, vocab_size=vocab_size)
print(tokenizer.get_vocab())

# 测试 tokenizer
sentence = "Let's test this tokenizer, café."
encoding = tokenizer.encode(sentence)
decoding = tokenizer.decode(encoding.ids)
print(encoding)
print(decoding)

# tokenizer dataset
encoded_dataset = tokenize_dataset(dataset, tokenizer)
# print(encoded_dataset)

# ==================
#  Dataloader
# ==================

def get_batch(data, batch_size, seq_len, device):
    """Generates a batch of input-output sequences from the data."""
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+seq_len], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+seq_len+1], dtype=torch.long) for i in ix])
    return x.to(device), y.to(device)

# ==================
#  Model: Transformer
# ==================

class MiniTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        # 多头自注意力层
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # 前馈全连接层
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        # 层归一化
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (seq_len, batch_size, embed_dim) —— nn.MultiheadAttention 要求这个顺序

        # ---- Multi-Head Self Attention ----
        x = self.ln1(x)                  # pre-norm
        attn_output, attn_weights = self.attn(x, x, x)
        x = x + self.dropout(attn_output)   # residual

        # ---- Feed Forward ----
        x = self.ln2(x)                  # pre-norm
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)     # residual

        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_heads=2, ff_hidden_dim=64, num_layers=2, max_seq_len=64):
        super().__init__()
        # token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 位置编码（可学或固定）
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        # Transformer 层堆叠
        self.layers = nn.ModuleList([
            MiniTransformerBlock(embed_dim, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ])
        # 分类头
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        # self.head.weight = self.embedding.weight  # tie embedding and output weights

    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x + self.pos_embedding[:seq_len, :]  # 广播位置编码
        x = x.transpose(0, 1)  # 转成 (seq_len, batch_size, embed_dim) 方便 nn.MultiheadAttention

        for layer in self.layers:
            x = layer(x)

        out = self.head(x)  # shape: (seq_len, batch_size, embed_dim)
        out = out.transpose(0, 1)  # (batch_size, seq_len, embid_dim)
        return out
    
# =======================
#  Training
# =======================

seq_len = 16
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if getattr(torch, "has_mps", False) and torch.backends.mps.is_available() else "cpu")


model = MiniTransformer(
    vocab_size=tokenizer.get_vocab_size(),
    embed_dim=64,
    num_heads=2,
    ff_hidden_dim=32,
    num_layers=4,
    max_seq_len=seq_len
)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
criterion = torch.nn.CrossEntropyLoss()

steps = 300
batch_size = 512
model.train()
for step in range(steps):
    xb, yb = get_batch(encoded_dataset, batch_size, seq_len, device=device)
    logits = model(xb)
    loss = criterion(
        logits.reshape(-1, tokenizer.get_vocab_size()), 
        yb.reshape(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    # Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step} | loss = {loss.item():.4f}")

# =======================
# Generation
# =======================

@torch.no_grad()
def generate(model, tokenizer, start_text="", max_new_tokens=20, device="cpu"):
    # Tokenize the prompt
    start_ids = [tokenizer.encode(start_text).ids] if start_text else []
    if start_ids:
        input_ids = torch.tensor(start_ids, dtype=torch.long, device=device)
    else:
        input_ids = torch.tensor([[tokenizer.token_to_id("<|endoftext|>")]], dtype=torch.long, device=device)

    generated = input_ids.tolist()[0]

    for _ in range(max_new_tokens):
        # Truncate to model max seq_len if needed
        idx_cond = input_ids[:, -model.pos_embedding.size(0):]

        # Forward pass
        logits = model(idx_cond)  # (batch_size=1, seq_len, vocab_size)
        logits_last = logits[:, -1, :]  # take last token logits

        # Greedy sampling
        # next_id = torch.argmax(logits_last, dim=-1).item()

        # Sample with probability
        probs = nn.functional.softmax(logits_last, dim=-1)  # (B, vocab_size)
        next_id = torch.multinomial(probs, num_samples=1)  # (B,1)

        # Append to sequence
        generated.append(next_id)
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)

        # Stop if EOS
        if next_id == tokenizer.token_to_id("<|endoftext|>"):
            break

    # Decode tokens
    return tokenizer.decode(generated)

# =======================
# Example usage
# =======================
model.eval()
prompt = "neural"
generated_text = generate(model, tokenizer, start_text=prompt, max_new_tokens=8, device=device)
print("Prompt:", prompt)
print("Generated:", generated_text)
