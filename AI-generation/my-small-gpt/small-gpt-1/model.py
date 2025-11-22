# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from config import GPTConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([x2, x1], dim=-1)

def apply_rope(x, sin, cos):
    # x: (B, n_heads, T, head_dim)
    return (x * cos) + (rotate_half(x) * sin)

def build_rope_cache(seq_len, head_dim, device):
    theta = 10000 ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    t = torch.arange(seq_len, device=device).float()
    
    sinusoid = torch.einsum("i,j->ij", t, theta)
    sin = sinusoid.sin().unsqueeze(0).unsqueeze(0)  # (1,1,T,head_dim/2)
    cos = sinusoid.cos().unsqueeze(0).unsqueeze(0)  # (1,1,T,head_dim/2)

    sin = torch.cat([sin, sin], dim=-1)  # repeat to match head_dim
    cos = torch.cat([cos, cos], dim=-1)
    return sin, cos


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
        self.register_buffer("rope_sin", None)
        self.register_buffer("rope_cos", None)
        self.rope_seq_len = config.seq_len
        sin, cos = build_rope_cache(self.rope_seq_len, self.head_dim, config.device)
        self.rope_sin = sin
        self.rope_cos = cos
        self.head_dim = head_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.size()

        if (self.rope_sin is None) or (T > self.rope_seq_len):
            sin, cos = build_rope_cache(T, self.head_dim, x.device)
            self.rope_sin = sin
            self.rope_cos = cos


        qkv = self.qkv(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each shape (B, T, n_heads, head_dim)
        
        # transpose so shape: (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to q, k
        sin = self.rope_sin[:, :, :T, :]
        cos = self.rope_cos[:, :, :T, :]
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)
        
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
#   Transformer Language Model
# -----------------------------
class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # tie embedding and output weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # This fixes your 124 loss problem
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx: (B, T) input token indices
        targets: optional (B, T) next‑token targets
        returns: logits (B, T, vocab_size), optional loss
        """
        idx = idx.long()
        B, T = idx.size()
        device = idx.device
        
        tok_emb = self.token_emb(idx) * math.sqrt(self.config.d_model)  # (B, T, d_model)

        x = tok_emb
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
        eos_id: int = None,
        temperature: float = 0.8,
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
            if next_token.item() == eos_id:
                break
        return idx
    