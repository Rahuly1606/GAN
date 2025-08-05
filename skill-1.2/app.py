from flask import Flask, render_template, request
from model import train_and_evaluate

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    accuracy = None
    weights_biases = None
    model_type = 'ANN'
    activation = 'relu'

    if request.method == 'POST':
        model_type = request.form['model_type']
        activation = request.form['activation']
        accuracy, weights_biases = train_and_evaluate(model_type, activation)

    return render_template('index.html',
                           accuracy=accuracy,
                           weights_biases=weights_biases,
                           model_type=model_type,
                           activation=activation)

if __name__ == '__main__':
    app.run(debug=True)
