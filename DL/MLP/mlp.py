import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# Step 1: Define Transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5, std=0.5
])

# Step 2: Download the MNIST dataset
full_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

train_size = int(0.6 * len(full_dataset))  # 60% for training
valid_size = int(0.2 * len(full_dataset))  # 20% for validation
test_size = len(full_dataset) - train_size - valid_size  # 20% for testing

train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])


# Step 3: Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4096, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

# Step 4: Define a simple Neural Network (for example)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Check whether GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 5: Initialize the model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()  # For multi-class classification
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Move data to GPU if available
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # After each epoch, validate the model
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    valid_loss = 0.0
    with torch.no_grad():  # No gradients needed for validation
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Train loss: {running_loss/len(train_loader):.4f}, Validation loss: {valid_loss/len(train_loader):.4f}.  Validation Accuracy: {val_accuracy:.2f}%")


# Step 7: Test the Model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # No gradients needed for testing
    for inputs, labels in test_loader:
        # Move data to GPU if available
        inputs, labels = inputs.to(device), labels.to(device) 

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test data: {100 * correct / total:.2f}%")
