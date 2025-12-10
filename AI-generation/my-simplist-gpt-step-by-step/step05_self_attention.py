import torch
import torch.nn as nn
import math

# --------------------
# Self-Attention
# --------------------
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        att = Q @ K.transpose(-2, -1) / math.sqrt(x.size(-1))
        att = att.softmax(dim=-1)
        return att @ V

# --------------------
# Demo
# --------------------
x = torch.randn(2, 8, 32)
sa = SelfAttention(32)
y = sa(x)

print("input:", x.shape)
print("output:", y.shape)
