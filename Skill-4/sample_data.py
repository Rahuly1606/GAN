import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.GTSRB(root='./data', split='train', transform=transform, download=True)
test_dataset = datasets.GTSRB(root='./data', split='test', transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

num_classes = len(np.unique([label for _, label in train_dataset]))

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1) # 3 channels -> 8 filters
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Conv layer 1
        x = self.pool(torch.relu(self.conv2(x)))  # Conv layer 2
        x = x.view(-1, 16 * 8 * 8)                # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def visualize_filters(conv_layer):
    filters = conv_layer.weight.data.clone()
    filters = (filters - filters.min()) / (filters.max() - filters.min()) # Normalize to 0-1
    fig, ax = plt.subplots(1, filters.shape[0], figsize=(15, 5))
    for i in range(filters.shape[0]):
        ax[i].imshow(filters[i].permute(1, 2, 0))
        ax[i].axis('off')
    plt.show()

def visualize_feature_maps(img, conv_layer):
    with torch.no_grad():
        feature_maps = conv_layer(img.unsqueeze(0))
        feature_maps = feature_maps.squeeze(0)
        fig, ax = plt.subplots(1, feature_maps.shape[0], figsize=(15, 5))
        for i in range(feature_maps.shape[0]):
            ax[i].imshow(feature_maps[i].cpu(), cmap='gray')
            ax[i].axis('off')
        plt.show()

for epoch in range(2):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/2], Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")

sample_img, _ = next(iter(train_loader))

plt.imshow(np.transpose(sample_img[0], (1, 2, 0)))
plt.title("Original Image")
plt.show()

print("Visualizing filters of first convolution layer:")
visualize_filters(model.conv1)

print("Visualizing feature maps after first conv layer:")
visualize_feature_maps(sample_img[0], model.conv1)

