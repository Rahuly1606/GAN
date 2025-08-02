# app.py
from flask import Flask, render_template, request
import torch
import torch.nn as nn

app = Flask(__name__)

class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

model = RegressionModel(8)
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    features = None
    if request.method == "POST":
        features = [float(request.form[f"f{i}"]) for i in range(8)]
        with torch.no_grad():
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            prediction = model(input_tensor).item()

    return render_template("index.html", prediction=prediction, features=features)

if __name__ == "__main__":
    app.run(debug=True)
