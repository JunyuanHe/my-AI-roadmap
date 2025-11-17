from datasets import load_dataset
from collections import Counter
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Step: load dataset

dataset = load_dataset("mikasenghaas/wikitext-2")

# Check the available splits (e.g., train, validation, test)
print(dataset)



# Step: preprocess the data

# Load the train data (the text from WikiText-2)
train_text = dataset["train"]["text"]

# Join the text into a single string
train_text = " ".join(train_text)

# Tokenize the text into words
words = train_text.split()


# Word frequency filtering to reduce vocab size

# Count word frequencies
word_counts = Counter(words)

# Filter: keep only words that appear >= min_freq
min_freq = 5  # you can adjust this
# filtered_words = {word for word, count in word_counts.items() if count >= min_freq}

filtered_words = sorted([w for w, c in word_counts.items() if c >= min_freq])
vocab = {word: idx for idx, word in enumerate(filtered_words)}
vocab["<UNK>"] = len(vocab)

# Build vocab with special tokens
# vocab = {word: idx for idx, word in enumerate(filtered_words)}
# vocab["<UNK>"] = len(vocab)   # unknown token
reverse_vocab = {idx: word for word, idx in vocab.items()}


print(f"Original vocab size: {len(word_counts)}")
print(f"Filtered vocab size: {len(vocab)}")

# Tokenize with unknown replacement
def encode_text(text):
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()]

# Encode the training text
tokenized_text = encode_text(train_text)
print(tokenized_text[:20])

# # Tokenize the entire text into indices
# def encode_text(text):
#     return [vocab[word] for word in text.split() if word in vocab]

# # Encode the training text
# tokenized_text = encode_text(train_text)

# # Check the first 20 tokens
# print(tokenized_text[:20])



# Step: Create dataset and dataloader to handle batching

class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq_in = torch.tensor(self.data[idx:idx + self.seq_length])
        seq_out = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1])
        return seq_in, seq_out

# Set sequence length (number of words the model looks at to predict the next word)
seq_length = 30

# Create the dataset
dataset = TextDataset(tokenized_text, seq_length)

# Create DataLoader instances for training
batch_size = 64
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)



# Step: Define the RNN model

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, num_layers, device):
        return torch.zeros(num_layers, batch_size, hidden_dim).to(device)

# Hyperparameters
embedding_dim = 64
hidden_dim = 64
num_layers = 1
learning_rate = 0.005
epochs = 1

# Select device as cuda if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize the model
model = RNNModel(vocab_size=len(vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader)}")

# Save the model weights
torch.save(model.state_dict(), "rnn-textgen.pth")



# Step: Text generation (Sampling)

def generate_text(model, num_layers, device, start_text, length=100):
    model.eval()
    tokens = encode_text(start_text)
    input_seq = torch.tensor(tokens, device=device).unsqueeze(0)  # Add batch dimension

    hidden = model.init_hidden(batch_size=1, num_layers=num_layers, device=device)  # Initialize hidden state for a batch size of 1
    
    generated = start_text

    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            predicted_idx = output.argmax(dim=-1)[:, -1].item()
            predicted_word = reverse_vocab[predicted_idx]
            generated += " " + predicted_word
            input_seq = torch.tensor([predicted_idx], device=device).unsqueeze(0)  # Update input

    return generated

# Generate text after training
start_text = "The quick brown fox"
generated_text = generate_text(model, num_layers, device, start_text, length=100)
print(generated_text)



# Step: Evaluate on the test set

# Evaluation loop (optional)
model.eval()
total_loss = 0
hidden = model.init_hidden(32, num_layers=num_layers, device=device)

for batch_idx, (inputs, targets) in enumerate(train_loader):
    # Evaluate without gradients
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))
        total_loss += loss.item()

print(f"Evaluation Loss: {total_loss/len(train_loader)}")
