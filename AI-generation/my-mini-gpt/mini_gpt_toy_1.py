import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random

# -----------------------------
# 1. Hyperparameters
# -----------------------------
vocab_size = 1000       # toy vocabulary
d_model = 128           # embedding dimension
n_heads = 4             # attention heads
n_layers = 4            # transformer layers
seq_len = 32            # max sequence length
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2. Simple tokenizer (toy)
# -----------------------------
# Here we just use integers as tokens for simplicity
def random_data(num_samples=1000):
    data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))
    # data:    [x0, x1, x2, x3, x4]
    # input:   [x0, x1, x2, x3]  all tokens except the last one
    # target:  [x1, x2, x3, x4]  all tokens except the first one
    return data[:, :-1], data[:, 1:]  # (input, target)

train_x, train_y = random_data()

# -----------------------------
# 3. Multi-Head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

# -----------------------------
# 4. FeedForward + Block
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# -----------------------------
# 5. Transformer Language Model
# -----------------------------
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -seq_len:]
            logits, _ = self(idx_cond)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

# -----------------------------
# 6. Training
# -----------------------------
model = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(200):  # small number of steps
    ix = torch.randint(0, train_x.size(0), (batch_size,))
    x, y = train_x[ix].to(device), train_y[ix].to(device)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"Step {step}: loss = {loss.item():.3f}")

# -----------------------------
# 7. Text Generation (toy)
# -----------------------------
context = torch.randint(0, vocab_size, (1, 8)).to(device)
print("Input:", context.tolist())
output = model.generate(context, max_new_tokens=20)
print("Generated:", output.tolist())
