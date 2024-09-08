from flask import Flask, render_template, request, jsonify
from main import train_crypto_model, predict_algorithm

app = Flask(__name__)

# File path for the CSV
file_path = "cleaned_crypto_dataset.csv"

# Train the model when the server starts
pipeline, label_encoder = train_crypto_model(file_path)

# Define the home route
@app.route('/')
def index():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ciphertext = data.get('ciphertext', '')
    result = predict_algorithm(pipeline, label_encoder, ciphertext)
    return jsonify({'algorithm': result})


if __name__ == '__main__':
    app.run(debug=True)
