import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step: Data preprocessing and loading CIFAR-10 dataset

# Define transformations for training and testing
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization (mean, std)
])

# Load training and test datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Step: define the CNN model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layers with ReLU activation and max-pooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Assuming input size is 32x32
        self.fc2 = nn.Linear(512, 10)  # 10 classes in CIFAR-10

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Max-pooling after conv1
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Max-pooling after conv2
        x = F.relu(F.max_pool2d(self.conv3(x), 2))  # Max-pooling after conv3
        x = x.view(-1, 128 * 4 * 4)  # Flatten the tensor before feeding to fully connected layer
        x = F.relu(self.fc1(x))  # Fully connected layer 1 with ReLU
        x = self.dropout(x)  # Dropout for regularization
        x = self.fc2(x)  # Output layer (logits for classification)
        return x


# Step: training the model

# Check whether GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model, loss function, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Function for training the model
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device) # move to GPU
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update weights
            
            running_loss += loss.item()  # Track the loss for each batch
            # if i % 100 == 99:  # Print every 100 batches
            #     print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100}')
            #     running_loss = 0.0
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')


# Step: Evaluating the model

# Function for evaluating the model
def evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device) # move to GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10,000 test images: {accuracy}%')


# Step: Do training and evaluation

# Train the model
train(model, train_loader, criterion, optimizer, epochs=10)

# Evaluate the model on the test set
evaluate(model, test_loader)

# save the model
torch.save(model.state_dict(), 'cnn_cifar10.pth')
