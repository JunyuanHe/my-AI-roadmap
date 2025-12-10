import torch
import torch.nn as nn

# --------------------
# Multihead Attention
# --------------------
attn = nn.MultiheadAttention(embed_dim=32, num_heads=4)

# (T, B, C)
x = torch.randn(6, 2, 32)
y, w = attn(x, x, x)

print("input:", x.shape)
print("output:", y.shape)
