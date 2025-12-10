import torch
import torch.nn as nn

# --------------------
# Setup
# --------------------
B, T = 2, 6
V = 100
C = 32

x = torch.randint(0, V, (B, T))

# --------------------
# Embeddings
# --------------------
tok_emb = nn.Embedding(V, C)
pos_emb = nn.Parameter(torch.randn(T, C))

out = tok_emb(x) + pos_emb

# --------------------
# Inspect
# --------------------
print("input shape:", x.shape)
print("output shape:", out.shape)
