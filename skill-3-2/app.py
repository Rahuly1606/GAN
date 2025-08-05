from flask import Flask, render_template, request
from model import train_custom_model
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        num_layers = int(request.form.get('num_layers'))
        dropout = float(request.form.get('dropout'))
        optimizer = request.form.get('optimizer')
        scheduler = request.form.get('scheduler') == 'yes'

        accuracy = train_custom_model(
            num_layers=num_layers,
            dropout=dropout,
            optimizer_name=optimizer,
            use_scheduler=scheduler
        )
        result = {"accuracy": accuracy}

    return render_template("index.html", result=result)

if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
