import json
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load logs
# -----------------------------
train_loss_path = "AI-generation/my-small-gpt/small-gpt-1/model/train_loss_log.json"
val_loss_path   = "AI-generation/my-small-gpt/small-gpt-1/model/val_loss_log.json"

with open(train_loss_path, "r") as f:
    train_loss = json.load(f)

with open(val_loss_path, "r") as f:
    val_loss = json.load(f)

# -----------------------------
# Step arrays
# -----------------------------
train_steps = np.arange(len(train_loss))
val_steps = np.linspace(0, len(train_loss)-1, len(val_loss))

# -----------------------------
# 1️⃣ Plot Loss
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(train_steps, train_loss, label="Train Loss")
plt.plot(val_steps, val_loss, label="Validation Loss", marker='o', linestyle='--')
plt.title("Training and Validation Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("AI-generation/my-small-gpt/small-gpt-1/model/loss_curve.pdf")
plt.savefig("AI-generation/my-small-gpt/small-gpt-1/model/loss_curve.png", dpi=300)
plt.close()  # 防止窗口弹出

# -----------------------------
# 2️⃣ Plot PPL
# -----------------------------
train_ppl = np.exp(train_loss)
val_ppl = np.exp(val_loss)

plt.figure(figsize=(10,4))
plt.plot(train_steps, train_ppl, label="Train PPL")
plt.plot(val_steps, val_ppl, label="Validation PPL", marker='o', linestyle='--')
plt.title("Training and Validation Perplexity (PPL)")
plt.xlabel("Step")
plt.ylabel("PPL")
plt.yscale("log")  # 可选对数刻度，更好显示初期大值
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
# plt.savefig("AI-generation/my-small-gpt/small-gpt-1/model/ppl_curve.pdf")
plt.savefig("AI-generation/my-small-gpt/small-gpt-1/model/ppl_curve.png", dpi=300)
plt.close()  # 防止窗口弹出

print("Loss and PPL curves saved as PNG.")
