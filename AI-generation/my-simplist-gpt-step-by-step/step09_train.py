import torch
import torch.nn as nn
import numpy as np

from step08_model import *
from step02_dataloader import *

data = np.random.randint(0, 100, (1000,))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

for step in range(100):
    x, y = get_batch(data, 32, 16)
    # print(x.shape)
    # print(y.shape)
    logits = model(x)
    # print(logits.shape)
    loss = criterion(logits.contiguous().view(-1, V), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(step, loss.item())
