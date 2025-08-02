# app.py
from flask import Flask, render_template, request, redirect, send_file
from model import train_model, predict_batch
import os
import tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Default model config
DEFAULT_ACTIVATION = 'ReLU'
DEFAULT_OPTIMIZER = 'Adam'
DEFAULT_HIDDEN_LAYERS = [32, 16]

# Initial training
result = train_model(hidden_layers=DEFAULT_HIDDEN_LAYERS,
                     activation_name=DEFAULT_ACTIVATION,
                     optimizer_name=DEFAULT_OPTIMIZER)

model = result['model']
scaler = result['scaler']
feature_names = result['feature_names']

@app.route("/", methods=["GET", "POST"])
def index():
    global model, scaler, result
    r2 = result['r2_score']
    selected_activation = DEFAULT_ACTIVATION
    selected_optimizer = DEFAULT_OPTIMIZER
    layers_str = ",".join(map(str, DEFAULT_HIDDEN_LAYERS))

    if request.method == "POST":
        selected_activation = request.form.get("activation", DEFAULT_ACTIVATION)
        selected_optimizer = request.form.get("optimizer", DEFAULT_OPTIMIZER)
        hidden_layers_input = request.form.get("hidden_layers", layers_str)
        try:
            layers_list = list(map(int, hidden_layers_input.split(",")))
        except:
            layers_list = DEFAULT_HIDDEN_LAYERS

        # Retrain with new config
        result = train_model(hidden_layers=layers_list,
                             activation_name=selected_activation,
                             optimizer_name=selected_optimizer)
        model = result['model']
        scaler = result['scaler']
        r2 = result['r2_score']
        layers_str = ",".join(map(str, layers_list))

    return render_template("index.html",
                           r2_score=r2,
                           activation=selected_activation,
                           optimizer=selected_optimizer,
                           hidden_layers=layers_str)

@app.route("/upload", methods=["POST"])
def upload():
    if 'csvfile' not in request.files:
        return redirect("/")
    file = request.files['csvfile']
    if file.filename == '':
        return redirect("/")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    output_path = predict_batch(model, scaler, filepath)
    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
