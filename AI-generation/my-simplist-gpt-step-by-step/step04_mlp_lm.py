import torch
import torch.nn as nn

class MLPLM(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, vocab_size)
        )

    def forward(self, x):
        x = self.emb(x)
        return self.ff(x)

model = MLPLM(100, 32)
x = torch.randint(0, 100, (4, 8))
logits = model(x)
print(logits.shape)
