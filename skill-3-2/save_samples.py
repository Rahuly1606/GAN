import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

os.makedirs("test_images", exist_ok=True)

transform = transforms.ToTensor()
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

for i in range(5):
    image, label = test_dataset[i]
    save_image(image, f"test_images/sample_{i}_label_{label}.png")

print("Sample test images saved in 'test_images/' folder.")
