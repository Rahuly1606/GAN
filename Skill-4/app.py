from flask import Flask, request, render_template
import torch
import torch.nn as nn
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)


with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


class ToxicClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ToxicClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

model = ToxicClassifier(input_dim=5000)
model.load_state_dict(torch.load("toxic_model.pth", map_location=torch.device('cpu')))
model.eval()

labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        text = request.form["comment"]
        X = vectorizer.transform([text]).toarray()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            output = model(X_tensor).numpy()[0]
        prediction = dict(zip(labels, [round(float(x), 2) for x in output]))
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
