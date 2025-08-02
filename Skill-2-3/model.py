import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

class FlexibleHousingModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation):
        super(FlexibleHousingModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def get_activation_function(name):
    return {
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh
    }.get(name, nn.ReLU)

def get_optimizer(name, model_params, lr=0.01):
    return {
        'Adam': optim.Adam,
        'SGD': optim.SGD
    }.get(name, optim.Adam)(model_params, lr=lr)

def train_model(hidden_layers=[32, 16], activation_name='ReLU', optimizer_name='Adam'):
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    activation = get_activation_function(activation_name)
    model = FlexibleHousingModel(8, hidden_layers, activation)
    optimizer = get_optimizer(optimizer_name, model.parameters())
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    os.makedirs("static", exist_ok=True)
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig("static/loss_curve.png")
    plt.close()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
        r2 = r2_score(y_test, y_pred)

        plt.figure()
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted Prices")
        plt.savefig("static/actual_vs_predicted.png")
        plt.close()

    torch.save(model.state_dict(), "static/housing_model.pth")

    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'r2_score': r2
    }

def predict_batch(model, scaler, file_path):
    df = pd.read_csv(file_path)
    input_scaled = scaler.transform(df.values)
    inputs = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        preds = model(inputs).numpy().flatten()
    df['PredictedPrice'] = preds * 100000
    output_path = "static/predictions.csv"
    df.to_csv(output_path, index=False)
    return output_path
