# app.py
from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import load_model, class_names

app = Flask(__name__)

model_path = "fashion_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"'{model_path}' not found. Please run 'model.py' first to train and save the model.")

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file:
                try:
                    image = Image.open(image_file).convert('L')
                    image = transform(image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(image)
                        _, predicted = torch.max(outputs, 1)
                        prediction = class_names[predicted.item()]
                except Exception as e:
                    prediction = f"Error processing image: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
