import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib
matplotlib.use('Agg')  # Avoid GUI backend for server
import matplotlib.pyplot as plt

def preprocess_data():
    data = load_iris()
    X = data.data
    y = data.target.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), \
           torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError("Unsupported activation function")

def build_model(model_type='ANN', activation='relu'):
    act_fn = get_activation(activation)

    if model_type == 'ANN':
        return nn.Sequential(
            nn.Linear(4, 10),
            act_fn,
            nn.Linear(10, 3),
            nn.Softmax(dim=1)
        )
    elif model_type == 'DNN':
        return nn.Sequential(
            nn.Linear(4, 16),
            act_fn,
            nn.Linear(16, 12),
            act_fn,
            nn.Linear(12, 8),
            act_fn,
            nn.Linear(8, 3),
            nn.Softmax(dim=1)
        )

def train_and_evaluate(model_type='ANN', activation='relu'):
    X_train, y_train, X_test, y_test = preprocess_data()
    model = build_model(model_type, activation)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, torch.argmax(y_train, dim=1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Save training loss plot
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig('static/plot.png')
    plt.close()

    # Evaluate accuracy
    with torch.no_grad():
        y_pred = model(X_test)
        correct = (torch.argmax(y_pred, dim=1) == torch.argmax(y_test, dim=1)).sum().item()
        total = y_test.size(0)
        accuracy = correct / total

    # Extract weights and biases
    weights_biases = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            weights_biases.append({
                'weights': layer.weight.detach().numpy().tolist(),
                'biases': layer.bias.detach().numpy().tolist()
            })

    return accuracy, weights_biases
