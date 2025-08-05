import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

def build_model(input_size, hidden_sizes, output_size, dropout):
    layers = [nn.Flatten()]
    for h in hidden_sizes:
        layers.append(nn.Linear(input_size, h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        input_size = h
    layers.append(nn.Linear(input_size, output_size))
    return nn.Sequential(*layers)

def get_optimizer(name, model_params, lr=0.001, weight_decay=1e-4):
    if name == 'adam':
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)

def train_custom_model(num_layers=2, dropout=0.3, optimizer_name='adam', use_scheduler=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    hidden_sizes = [128] * num_layers
    model = build_model(28*28, hidden_sizes, 10, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters())

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) if use_scheduler else None

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience, patience_counter = 5, 0

    for epoch in range(50):
        model.train()
        running_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "fashion_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/loss_curve.png')

    model.load_state_dict(torch.load("fashion_model.pth"))
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, pred = torch.max(outputs, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return round(100 * correct / total, 2)
