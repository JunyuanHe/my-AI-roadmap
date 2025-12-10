import torch
import torch.nn as nn

# ----- tiny toy vocabulary -----
vocab = list("abcdefghijklmnopqrstuvwxyz ")
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
V = len(vocab)

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

# ----- dataset -----
text = "hello world goodbye world"
data = torch.tensor(encode(text), dtype=torch.long)

def get_batch(data, batch_size):
    ix = torch.randint(0, len(data) - 1, (batch_size,))
    x = torch.stack([data[i] for i in ix])
    y = torch.stack([data[i+1] for i in ix])
    return x, y

# ----- model -----
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.emb(x)

model = BigramLM(V)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

# ----- training -----
for step in range(200):
    x, y = get_batch(data, 32)
    logits = model(x)
    loss = loss_fn(logits.view(-1, V), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step:3d}:  loss = {loss.item()}")

# ----- generation -----
idx = torch.tensor([[stoi["h"]]])
for _ in range(20):
    logits = model(idx)
    next_id = torch.argmax(logits[:, -1], dim=-1)
    idx = torch.cat([idx, next_id[:, None]], dim=1)

print(decode(idx[0].tolist()))
