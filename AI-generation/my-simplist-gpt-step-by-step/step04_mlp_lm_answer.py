import torch
import torch.nn as nn
import numpy as np

# --------------------
# Data
# --------------------
data = np.random.randint(0, 100, (1000,))

# def get_batch(data, batch_size, seq_len):
#     ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
#     x = torch.stack([data[i:i+seq_len] for i in ix])
#     y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
#     return x, y

def get_batch(data, batch_size, seq_len):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+seq_len], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+seq_len+1], dtype=torch.long) for i in ix])
    return x, y

# --------------------
# Model
# --------------------
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

model = MLPLM(vocab_size=100, embed_dim=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

# --------------------
# Training
# --------------------
for step in range(200):
    x, y = get_batch(data, 32, 8)
    logits = model(x)
    loss = loss_fn(logits.view(-1, 100), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(step, loss.item())
