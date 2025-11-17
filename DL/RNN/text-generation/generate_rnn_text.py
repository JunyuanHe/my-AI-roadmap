# generate_rnn_text.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import argparse

# -------------------------------
# RNN model definition (must match the original)
# -------------------------------
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, embedding_weights=None):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
            self.embedding.weight.requires_grad = False
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        if hidden_dim == embedding_dim:
            self.fc.weight = self.embedding.weight
            self.fc.weight.requires_grad = False

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, num_layers, device):
        return torch.zeros(num_layers, batch_size, hidden_dim).to(device)

# -------------------------------
# Load tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
vocab_size = tokenizer.vocab_size
embedding_dim = 768  # GPT-2 default embedding dim
hidden_dim = embedding_dim
num_layers = 1

# -------------------------------
# Load pre-trained GPT-2 embeddings
# -------------------------------
from transformers import AutoModelForCausalLM
hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
hf_embeddings = hf_model.transformer.wte.weight
embedding_matrix = hf_embeddings.detach().clone()
embedding_matrix.requires_grad = False

# -------------------------------
# Device setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Initialize RNN and load weights
# -------------------------------
model = RNNModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    embedding_weights=embedding_matrix
).to(device)

# Load saved weights
model.load_state_dict(torch.load("rnn-textgen.pth", map_location=device))
model.eval()

# -------------------------------
# Text generation function
# -------------------------------
def generate_text(model, num_layers, device, start_text, length=300, use_sampling=False):
    model.eval()
    input_ids = tokenizer.encode(start_text, add_special_tokens=False, return_tensors="pt").to(device)
    hidden = model.init_hidden(batch_size=1, num_layers=num_layers, device=device)
    generated_ids = input_ids.squeeze(0).tolist()
    
    for _ in range(length):
        with torch.no_grad():
            outputs, hidden = model(input_ids, hidden)
            logits = outputs[:, -1, :]
            if use_sampling:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_id = torch.argmax(logits, dim=-1).item()
            generated_ids.append(next_id)
            input_ids = torch.tensor([[next_id]], device=device)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# -------------------------------
# Command line interface
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Starting text for generation")
    parser.add_argument("--length", type=int, default=300, help="Number of tokens to generate")
    parser.add_argument("--sampling", action="store_true", help="Use sampling instead of greedy decoding")
    args = parser.parse_args()

    generated_text = generate_text(model, num_layers, device, args.prompt, length=args.length, use_sampling=args.sampling)
    print("\n=== Generated Text ===\n")
    print(generated_text)
