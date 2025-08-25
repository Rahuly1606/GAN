import torch
import torchvision.transforms as transforms
from flask import Flask, request, render_template
from PIL import Image
from model import CNNModel

app = Flask(__name__)

# Load model
model = CNNModel(num_attributes=3)
model.load_state_dict(torch.load("face_attributes.pth", map_location=torch.device("cpu")))
model.eval()

# Labels
attributes = ["Smiling", "Eyeglasses", "Male"]

# Transform for input image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = Image.open(file).convert("RGB")
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)[0]
            result = {attributes[i]: "Yes" if outputs[i] > 0.5 else "No"
                      for i in range(len(attributes))}
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

