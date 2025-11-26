# train.py

import torch
import os
import math
import json
from torch.optim.lr_scheduler import LambdaLR
from model import MiniGPT
from config import GPTConfig
from tokenizer import MyTokenizer
from datasets import load_dataset
from utils import get_batch
from torchsummary import summary
import numpy as np

torch.set_float32_matmul_precision("high")



# Tokenize

# Load the trained tokenizer
tokenizer_pth = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer-tinystoriesv2.json"
# tokenizer_pth = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer-simplebooks.json"
# tokenizer_pth = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer.json"
tokenizer = MyTokenizer(tokenizer_pth)  # Make sure this points to the directory with tokenizer files
config = GPTConfig(vocab_size=tokenizer.vocab_size)

# Ensure the directory exists
os.makedirs(config.model_dir, exist_ok=True)

# Load dataset
train_path = os.path.join(config.data_dir, "train-simplebooks.bin")
val_path = os.path.join(config.data_dir, "val-simplebooks.bin")
# train_path = os.path.join(config.data_dir, "train.bin")
# val_path = os.path.join(config.data_dir, "val.bin")
token_ids = np.memmap(train_path, dtype=np.uint16, mode='r')
val_ids = np.memmap(val_path, dtype=np.uint16, mode='r')


# Convert the text to token IDs using the tokenizer
# token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Setup model and training loop as before
device = config.device



def lr_lambda(step):
    if step < config.warmup_steps:
        return step / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def estimate_loss(model, token_ids, batch_size, seq_len, config, eval_iters=100):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            xb, yb = get_batch(token_ids, batch_size, seq_len, config)
            logits, loss = model(xb, yb)
            losses.append(loss.item())
    model.train()
    return sum(losses)/len(losses)


model = MiniGPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, betas=(0.9, 0.95))

# scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=config.max_steps - config.warmup_steps,
    eta_min=config.min_lr
)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer,
#     T_0=500,       # number of steps in the first restart
#     T_mult=2,      # multiply T_i by this after each restart
#     eta_min=config.min_lr # minimum learning rate
# )

# scheduler = LambdaLR(optimizer, lr_lambda)



steps = config.max_steps
batch_size = 64
seq_len = config.seq_len

summary(model, input_size=(seq_len,))  # Only specify (seq_len,) here

# Compile model
model = torch.compile(model) # requires PyTorch 2.0


train_loss_log = []
val_loss_log = []
best_val_loss = float('inf')

# Training loop
for step in range(steps):
    xb, yb = get_batch(token_ids, batch_size, seq_len, config)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    # Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    # lr schedule: warmup + cosine restart
    if step < config.warmup_steps:
        lr = config.max_lr * step / config.warmup_steps
        for g in optimizer.param_groups:
            g["lr"] = lr
    else:
        scheduler.step()

    train_loss_log.append(float(loss.item()))

    if step % 100 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        val_loss = estimate_loss(model, val_ids, batch_size, seq_len, config, eval_iters=20)
        val_loss_log.append(float(val_loss))
        print(f"Step {step} | LR = {current_lr:.6f} | loss = {loss.item():.4f} | val_loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config.model_dir, "small_gpt_1_best.pth")
            # torch.save(model.state_dict(), save_path)
            torch.save({
                'step': step,
                'model_state_dict': model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"\tBest model saved at step {step}, val_loss={val_loss:.4f}")


# Save the model after training

save_path = os.path.join(config.model_dir, "small_gpt_1_last.pth")

torch.save({
    'step': steps,
    'model_state_dict': model._orig_mod.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, save_path)
print(f"Last model saved.")

# -----------------------------
# Save final logs
# -----------------------------
os.makedirs(config.model_dir, exist_ok=True)
with open(os.path.join(config.model_dir, "train_loss_log.json"), "w") as f:
    json.dump(train_loss_log, f)
with open(os.path.join(config.model_dir, "val_loss_log.json"), "w") as f:
    json.dump(val_loss_log, f)
print("Training and validation loss logs saved.")

# # Save loss log
# loss_log_path = os.path.join(config.model_dir, "loss_log.json")
# # "AI-generation/my-small-gpt/small-gpt-1/model/loss_log.json"
# with open(loss_log_path, "w") as f:
#     json.dump(loss_log, f)
# print(f"Loss log saved to {loss_log_path}")