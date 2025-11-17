"""
Optimized in terms of code style. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# -----------------------------
# 1. Configuration / Hyperparams
# -----------------------------
class GPTConfig:
    def __init__(
        self,
        vocab_size: int = 1000,
        seq_len: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

# -----------------------------
# 2. Modules: Attention, FeedForward, Block
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        d_model = config.d_model
        n_heads = config.n_heads
        head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # register a causal mask buffer so it can be reused
        mask = torch.tril(torch.ones(config.seq_len, config.seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each shape (B, T, n_heads, head_dim)
        
        # transpose so shape: (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, T, T)
        # apply causal mask
        mask = self.causal_mask[:T, :T]
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        attn_out = attn_probs @ v  # (B, n_heads, T, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        out = self.proj(attn_out)
        out = self.resid_dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        d_model = config.d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(config.dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        d_model = config.d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# -----------------------------
# 3. Transformer Language Model
# -----------------------------
class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx: (B, T) input token indices
        targets: optional (B, T) next‑token targets
        returns: logits (B, T, vocab_size), optional loss
        """
        B, T = idx.size()
        device = idx.device
        
        tok_emb = self.token_emb(idx)  # (B, T, d_model)
        pos_idx = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_emb(pos_idx)  # (1, T, d_model)
        
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # flatten for cross‑entropy: (B*T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        idx: (B, T) initial prompt
        returns: (B, T + max_new_tokens) with generated tokens appended
        """
        for _ in range(max_new_tokens):
            B, T = idx.size()
            if T > self.config.seq_len:
                idx_cond = idx[:, -self.config.seq_len :]
            else:
                idx_cond = idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                min_topk = v[:, -1].unsqueeze(1)
                logits = torch.where(logits < min_topk, torch.full_like(logits, float("-inf")), logits)
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, next_token), dim=1)  # (B, T+1)
        return idx

# -----------------------------
# 4. Data & Training Loop (toy)
# -----------------------------
def random_data(num_samples: int, config: GPTConfig):
    data = torch.randint(0, config.vocab_size, (num_samples, config.seq_len + 1))
    return data[:, :-1], data[:, 1:]

def train_toy():
    config = GPTConfig()
    device = config.device
    train_x, train_y = random_data(1000, config)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    steps = 200
    batch_size = 16
    for step in range(steps):
        ix = torch.randint(0, train_x.size(0), (batch_size,), device=device)
        xb = train_x[ix]
        yb = train_y[ix]
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")
    
    # generation demo
    context = torch.randint(0, config.vocab_size, (1, 8), device=device)
    print("Input:", context.tolist())
    out = model.generate(context, max_new_tokens=20, temperature=1.0, top_k=5)
    print("Generated:", out.tolist())

if __name__ == "__main__":
    train_toy()
