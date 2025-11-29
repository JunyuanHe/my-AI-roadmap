# import torch
# import torch.nn as nn

# # ==================
# #  Tokenizer
# # ==================

# from tokenizers import Tokenizer, Regex
# from tokenizers.models import BPE
# from tokenizers.trainers import BpeTrainer
# from tokenizers.pre_tokenizers import Split

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# def build_tokenizer(texts, vocab_size=2000):
#     tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
#     tokenizer.pre_tokenizer = Split(Regex(PAT), behavior="isolated")
#     trainer = BpeTrainer(
#         vocab_size=vocab_size,
#         special_tokens=["[UNK]", "<|endoftext|>"]
#     )
#     tokenizer.train_from_iterator(texts, trainer)
#     return tokenizer


# # ==================
# #  Dataset
# # ==================

# from datasets import Dataset
# import torch

# def tokenize_dataset(dataset, tokenizer, max_len=32):

#     def encode(batch):
#         ids = tokenizer.encode(batch["text"]).ids
#         ids = ids[:max_len]
#         # padding
#         if len(ids) < max_len:
#             ids = ids + [tokenizer.token_to_id("[PAD]")]*(max_len-len(ids))
#         batch["input_ids"] = ids
#         return batch

#     return dataset.map(encode)


# # Example tiny dataset
# texts = [
#     "hello world",
#     "machine learning is fun",
#     "transformers are powerful",
#     "attention is all you need",
# ]

# dataset = Dataset.from_dict({"text": texts})

# # Train tokenizer
# tokenizer = build_tokenizer(texts, vocab_size=50)
# print(tokenizer.get_vocab())
# # tokenizer dataset
# encoded_dataset = tokenize_dataset(dataset, tokenizer)
# print(encoded_dataset)

# # ==================
# #  Dataloader
# # ==================

# def collate_fn(batch):
#     ids = [torch.tensor(item["input_ids"]) for item in batch]
#     return torch.stack(ids)   # (batch, seq_len)

# from torch.utils.data import DataLoader
# loader = DataLoader(encoded_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
# print(loader)




# # ==================
# #  Model: Transformer
# # ==================

# class MiniTransformerBlock(nn.Module):
#     def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
#         super().__init__()
#         # 多头自注意力层
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         # 前馈全连接层
#         self.ff = nn.Sequential(
#             nn.Linear(embed_dim, ff_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(ff_hidden_dim, embed_dim)
#         )
#         # 层归一化
#         self.ln1 = nn.LayerNorm(embed_dim)
#         self.ln2 = nn.LayerNorm(embed_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # x: (seq_len, batch_size, embed_dim) —— nn.MultiheadAttention 要求这个顺序

#         # ---- Multi-Head Self Attention ----
#         x = self.ln1(x)                  # pre-norm
#         attn_output, attn_weights = self.attn(x, x, x)
#         x = x + self.dropout(attn_output)   # residual

#         # ---- Feed Forward ----
#         x = self.ln2(x)                  # pre-norm
#         ff_output = self.ff(x)
#         x = x + self.dropout(ff_output)     # residual

#         return x

# class MiniTransformer(nn.Module):
#     def __init__(self, vocab_size, embed_dim=32, num_heads=2, ff_hidden_dim=64, num_layers=2, max_len=50, num_classes=10):
#         super().__init__()
#         # token embedding
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         # 位置编码（可学或固定）
#         self.pos_embedding = nn.Parameter(torch.randn(max_len, embed_dim))
#         # Transformer 层堆叠
#         self.layers = nn.ModuleList([
#             MiniTransformerBlock(embed_dim, num_heads, ff_hidden_dim)
#             for _ in range(num_layers)
#         ])
#         # 分类头
#         self.head = nn.Linear(embed_dim, vocab_size)

#     def forward(self, x):
#         # x: (batch_size, seq_len)
#         batch_size, seq_len = x.size()
#         x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
#         x = x + self.pos_embedding[:seq_len, :]  # 广播位置编码
#         x = x.transpose(0, 1)  # 转成 (seq_len, batch_size, embed_dim) 方便 nn.MultiheadAttention

#         for layer in self.layers:
#             x = layer(x)

#         x = x.mean(dim=0)  # 对序列维度做池化
#         out = self.classifier(x)
#         return out
    
# # =======================
# #  Training
# # =======================

# model = MiniTransformer(
#     vocab_size=tokenizer.get_vocab_size(),
#     embed_dim=64,
#     num_heads=4,
#     ff_hidden_dim=128,
#     num_layers=2,
#     max_len=32
# )

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = torch.nn.CrossEntropyLoss()

# num_epochs = 10
# model.train()
# for epoch in range(num_epochs):
#     for batch in loader:
        
#         input_ids = batch          # (B, T)
#         labels = batch             # language modeling: predict next token
        
#         # shift targets to the right: input_ids[:, 0..T-2] → labels[:, 1..T-1]
#         labels = labels[:, 1:].contiguous()
#         inputs = input_ids[:, :-1].contiguous()

#         # forward pass
#         logits, _ = model(inputs)      # outputs: (B, T-1, vocab)
        
#         # reshape for loss
#         loss = criterion(
#             logits.view(-1, vocab_size),
#             labels.reshape(-1)
#         )
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch} | Loss: {loss.item():.4f}")




# # =======================
# # 快速测试
# # =======================
# if __name__ == "__main__":
#     vocab_size = 100
#     seq_len = 10
#     batch_size = 4
#     num_classes = 5

#     model = MiniTransformer(vocab_size=vocab_size, num_classes=num_classes)
#     sample_input = torch.randint(0, vocab_size, (batch_size, seq_len))
#     output = model(sample_input)
#     print("Output shape:", output.shape)  # (batch_size, num_classes)
