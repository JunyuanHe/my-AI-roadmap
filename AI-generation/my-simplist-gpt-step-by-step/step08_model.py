import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab, dim, heads, layers, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(max_len, dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads)
            for _ in range(layers)
        ])
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        B, T = x.shape
        x = self.emb(x) + self.pos[:T]
        x = x.transpose(0, 1)
        for block in self.blocks:
            x = block(x)
        return self.head(x).transpose(0, 1)

# --------------------
# Demo
# --------------------
V = 100
model = Transformer(V, 32, 4, 2, 64)
x = torch.randint(0, V, (2, 16))
logits = model(x)

print("logits:", logits.shape)