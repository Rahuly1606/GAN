import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

# Set random seed
torch.manual_seed(0)

# Load and preprocess data
iris = load_iris()
X = iris.data  # shape: (150, 4)
y = iris.target.reshape(-1, 1)  # shape: (150, 1)

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)  # fixed for sklearn >= 1.2
y_encoded = encoder.fit_transform(y)

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to torch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.float32)  # One-hot labels for softmax output

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Define ANN model with 1 hidden layer
class IrisANN(nn.Module):
    def __init__(self):
        super(IrisANN, self).__init__()
        self.hidden = nn.Linear(4, 10)   # hidden layer with 10 neurons
        self.output = nn.Linear(10, 3)   # output layer with 3 classes

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.softmax(self.output(x), dim=1)
        return x

# Instantiate the model
model = IrisANN()

# Define loss function and optimizer
criterion = nn.MSELoss()  # using one-hot + softmax, so use MSELoss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
loss_list = []
epochs = 100

for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = torch.argmax(predictions, dim=1)
    actual_classes = torch.argmax(y_test, dim=1)
    accuracy = (predicted_classes == actual_classes).float().mean().item()

# Display weights and biases
print("\n--- Weights and Biases ---")
for name, param in model.named_parameters():
    print(f"{name}:\n{param.data.numpy()}\n")

# Plot training loss curve
plt.plot(loss_list)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Final test accuracy
print(f"\nFinal Test Accuracy: {accuracy:.4f}")
