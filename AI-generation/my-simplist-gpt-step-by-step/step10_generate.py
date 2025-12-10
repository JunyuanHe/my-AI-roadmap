import torch
import torch.nn as nn
import torch.nn.functional as F


from step09_train import *


# --------------------
# Generation
# --------------------

### max_probablity generation
@torch.no_grad()
def generate(model, start, max_new=20):
    idx = start
    for _ in range(max_new):
        logits = model(idx)
        print(logits[:, -1].shape)
        next_id = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx

### multinomial generation
# @torch.no_grad()
# def generate(model, start, max_new=20):
#     idx = start
#     for _ in range(max_new):
#         logits = model(idx)                 # (B, T, V)
#         logits_last = logits[:, -1, :]      # (B, V)

#         probs = F.softmax(logits_last, dim=-1)  # probabilities
#         next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

#         idx = torch.cat([idx, next_id], dim=1)
#     return idx


### multinomial generation with temperature
# @torch.no_grad()
# def generate(model, start, max_new=20, temperature=1.0):
#     idx = start
#     for _ in range(max_new):
#         logits = model(idx)
#         logits_last = logits[:, -1, :] / temperature

#         probs = F.softmax(logits_last, dim=-1)
#         next_id = torch.multinomial(probs, num_samples=1)

#         idx = torch.cat([idx, next_id], dim=1)
#     return idx


# --------------------
# Demo
# --------------------
start = torch.tensor([[0, 17, 9]])
out = generate(model, start, 10)
print("generated:", out)