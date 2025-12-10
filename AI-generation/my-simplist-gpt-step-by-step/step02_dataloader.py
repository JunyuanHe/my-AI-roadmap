import torch

# --------------------
# Fake tokenized corpus
# --------------------
tokens = list(range(100))  # pretend tokenized corpus

# --------------------
# Dataset construction
# --------------------
def get_batch(data, batch_size, seq_len):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+seq_len], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+seq_len+1], dtype=torch.long) for i in ix])
    return x, y

# --------------------
# Demo
# --------------------
x, y = get_batch(tokens, batch_size=4, seq_len=8)
print("x:", x)
print("y:", y)
