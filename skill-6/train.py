import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import CNNModel

batch_size = 64
epochs = 2   # Increase to ~10–20 for better accuracy
learning_rate = 0.001
attributes_to_use = ["Smiling", "Eyeglasses", "Male"]


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


dataset = datasets.CelebA(root="./data", split="train",
                          target_type="attr", transform=transform, download=True)


attr_idx = [dataset.attr_names.index(a) for a in attributes_to_use]


class CelebASubset(torch.utils.data.Dataset):
    def __init__(self, dataset, attr_idx):
        self.dataset = dataset
        self.attr_idx = attr_idx
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, attrs = self.dataset[idx]
        return img, attrs[self.attr_idx].float()

train_loader = DataLoader(CelebASubset(dataset, attr_idx), batch_size=batch_size, shuffle=True)


model = CNNModel(num_attributes=len(attributes_to_use))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")


torch.save(model.state_dict(), "face_attributes.pth")
print(" Training complete. Model saved as face_attributes.pth")

