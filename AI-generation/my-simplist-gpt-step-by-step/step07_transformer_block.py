import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ff(self.ln2(x))
        return x

# --------------------
# Demo
# --------------------
x = torch.randn(6, 2, 32)
block = TransformerBlock(32, 4)
y = block(x)

print("output:", y.shape)
