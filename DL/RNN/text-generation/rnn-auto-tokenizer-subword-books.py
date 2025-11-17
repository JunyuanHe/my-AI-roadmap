from datasets import load_dataset
from collections import Counter
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import re

def preprocess_text(text):
    """
    Cleans text for training a text generation model.
    
    Steps:
    1. Replace single newlines inside paragraphs with a space.
    2. Replace multiple newlines with a single newline (paragraph break).
    3. Normalize whitespace.
    """
    
    # Step 1: Replace single newlines (not double) with a space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Step 2: Replace multiple newlines with a single newline
    text = re.sub(r'\n{2,}', '\n', text)
    
    # Step 3: Normalize multiple spaces to a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Optional Step 4: Strip leading/trailing spaces on each line
    text = "\n".join(line.strip() for line in text.splitlines())
    
    return text


# Use GPT-2 tokenizer (BPE subwords)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # add PAD if needed
vocab_size = tokenizer.vocab_size


# Step: load dataset

# dataset = load_dataset("mikasenghaas/wikitext-2")

# # Check the available splits (e.g., train, validation, test)
# print(dataset)

# 1. Load a small book dataset
books = load_dataset("IsmaelMousa/books", split="train")

# 2. Sample a few books (say 10) to keep things lightweight
sample_books = books.shuffle(seed=40).select(range(1))
texts = sample_books["EN"]
print(f"Using {len(texts)} books for training.")

# Create token sequence
chunk_size = 4000
token_ids = []
for text in texts:
    for i in range(0, min(len(text), 100*chunk_size), chunk_size):
        chunk = text[i:i+chunk_size]
        chunk = preprocess_text(chunk)
        ids = tokenizer.encode(chunk, add_special_tokens=False)
        token_ids.extend(ids)  # cut each for memory if needed
        # token_ids.extend(ids[:50000])  # cut each for memory if needed

train_tokens = torch.tensor(token_ids)
print(f"Total subword tokens collected: {len(train_tokens)}")


# # Example: tokenize training text
# # train_text = " ".join(dataset["train"]["text"])
# print(len(dataset["train"]["text"]))

# # Use only the first N lines from training set
# N = 500  # adjust depending on memory
# subset_train_text = dataset["train"]["text"][:N]

# train_text = " ".join(subset_train_text)


# chunk_size = 2048  # or any size smaller than memory limit
# token_ids_list = []

# for i in range(0, len(train_text), chunk_size):
#     chunk = train_text[i:i+chunk_size]
#     tokens = tokenizer(chunk, add_special_tokens=False)["input_ids"]
#     token_ids_list.extend(tokens)

# train_tokens = torch.tensor(token_ids_list)
# print(len(train_tokens))


# Use GPT-2 tokenizer and model for embeddings
hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
hf_embeddings = hf_model.transformer.wte.weight  # embedding matrix
embedding_dim = hf_embeddings.shape[1]

# Initialize your embedding matrix for the RNN
embedding_matrix = hf_embeddings.detach().clone()

# freeze initially
embedding_matrix.requires_grad = False


# Step: Create dataset and dataloader to handle batching

class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq_in = self.data[idx:idx + self.seq_length]
        seq_out = self.data[idx + 1:idx + self.seq_length + 1]
        return seq_in, seq_out

# Set sequence length (number of words the model looks at to predict the next word)
seq_length = 50

# Create the dataset
dataset = TextDataset(train_tokens, seq_length)

# Create DataLoader instances for training
batch_size = 64
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)



# Step: Define the RNN model

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, embedding_weights=None):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
            # Optionally freeze embeddings at first:
            self.embedding.weight.requires_grad = False
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # If hidden_dim == embedding_dim, we can tie weights
        self.fc = nn.Linear(hidden_dim, vocab_size)
        if hidden_dim == embedding_dim:
            self.fc.weight = self.embedding.weight  # weight tying
            # Optionally freeze output layer at first:
            self.fc.weight.requires_grad = False

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, num_layers, device):
        return torch.zeros(num_layers, batch_size, hidden_dim).to(device)

# Hyperparameters
# embedding_dim = 64   # embedding_dim match pre-trained embedding
# hidden_dim = 64
hidden_dim = embedding_dim
num_layers = 1
learning_rate = 2e-3
epochs = 20

# Select device as cuda if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Initialize the model
model = RNNModel(
    vocab_size=vocab_size, 
    embedding_dim=embedding_dim, 
    hidden_dim=hidden_dim, 
    num_layers=num_layers, 
    embedding_weights=embedding_matrix
    ).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.8)  # lr = lr * 0.5 every 5 epochs
criterion = nn.CrossEntropyLoss()

# Print model architecture
print(model)


# Step: Train the model

for epoch in range(epochs):
    model.train()
    total_loss = 0
    hidden = model.init_hidden(batch_size=batch_size, num_layers=num_layers, device=device)  # Initialize hidden state for each batch
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        # Get the actual batch size for this batch
        batch_size = inputs.size(0)
        
        # Initialize the hidden state for this batch
        # hidden = model.init_hidden(batch_size=batch_size, num_layers=num_layers, device=device)
        
        # Move inputs and targets to GPU if available
        inputs, targets = inputs.to(device), targets.to(device)

        # Detach hidden state to prevent backpropagating through the entire history
        hidden = hidden.detach()

        # Forward pass
        outputs, hidden = model(inputs, hidden)

        # Compute loss
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader)}")

# Save the model weights
torch.save(model.state_dict(), "rnn-textgen.pth")



# Step: Text generation (Sampling)

def generate_text(model, num_layers, device, start_text, length=500, use_sampling=False):
    model.eval()
    input_ids = tokenizer.encode(start_text, add_special_tokens=False, return_tensors="pt").to(device)
    # input_seq = torch.tensor(tokens, device=device).unsqueeze(0)  # Add batch dimension

    hidden = model.init_hidden(batch_size=1, num_layers=num_layers, device=device)  # Initialize hidden state for a batch size of 1
    
    generated_ids = input_ids.squeeze(0).tolist()  # keep history
    
    for _ in range(length):
        with torch.no_grad():
            outputs, hidden = model(input_ids, hidden)   # shape: (1, seq_len, vocab_size)
            
            logits = outputs[:, -1, :]  # only the last step
            if use_sampling:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_id = torch.argmax(logits, dim=-1).item()
            
            generated_ids.append(next_id)
            
            # prepare next input (just the predicted token)
            input_ids = torch.tensor([[next_id]], device=device)
    
    # Use tokenizer.decode to cleanly convert tokens → text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

# Generate text after training
# start_text = "The quick brown fox"
start_text = "Time had forgotten the town"
generated_text = generate_text(model, num_layers, device, start_text, length=300)
print(generated_text)



# Step: Evaluate on the test set

# Evaluation loop (optional)
model.eval()
total_loss = 0
hidden = model.init_hidden(batch_size=batch_size, num_layers=num_layers, device=device)

for batch_idx, (inputs, targets) in enumerate(train_loader):
    # Evaluate without gradients
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        total_loss += loss.item()

print(f"Evaluation Loss: {total_loss/len(train_loader)}")
