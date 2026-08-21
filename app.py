import os
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense

app = Flask(__name__)

# Reconstruct Model Architecture matching config.json (10 Inputs -> 8 -> 7 -> 1)
def load_model():
    model = Sequential([
        InputLayer(shape=(10,), name="input_layer"),
        Dense(8, activation="relu", name="dense"),
        Dense(7, activation="relu", name="dense_1"),
        Dense(1, activation="sigmoid", name="dense_2")
    ])
    
    weights_path = os.path.join(os.path.dirname(__file__), "model.weights.h5")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    return model

model = load_model()

# Frontend HTML template with modern dark-mode styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Inference Dashboard</title>
    <style>
        body { background-color: #0e1117; color: #ffffff; font-family: -apple-system, sans-serif; padding: 2rem; }
        .container { max-width: 900px; margin: 0 auto; background: #1e222d; padding: 2rem; border-radius: 12px; border: 1px solid #2e364f; }
        h1 { color: #4CAF50; margin-top: 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
        .input-group { display: flex; flex-direction: column; }
        label { font-size: 0.85rem; color: #a0a0a0; margin-bottom: 0.3rem; }
        input { background: #0e1117; border: 1px solid #2e364f; color: white; padding: 0.5rem; border-radius: 6px; }
        button { width: 100%; background: #4CAF50; color: white; border: none; padding: 0.8rem; font-size: 1rem; font-weight: bold; border-radius: 6px; cursor: pointer; }
        button:hover { background: #45a049; }
        .result-box { margin-top: 1.5rem; padding: 1rem; background: #0e1117; border-radius: 8px; display: none; }
        .metric { font-size: 1.5rem; font-weight: bold; color: #4CAF50; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Neural Network Inference Engine</h1>
        <p>Enter values for the 10 input feature vectors:</p>
        <form id="predictForm">
            <div class="grid">
                {% for i in range(1, 11) %}
                <div class="input-group">
                    <label>Feature X_{{ i }}</label>
                    <input type="number" step="0.01" name="x{{ i }}" value="0.00" required>
                </div>
                {% endfor %}
            </div>
            <button type="submit">Run Prediction</button>
        </form>
        
        <div id="result" class="result-box">
            <h3>Prediction Results:</h3>
            <p>Predicted Probability: <span id="prob" class="metric">--</span></p>
            <p>Classification Outcome: <span id="class" class="metric">--</span></p>
        </div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const features = [];
            for (let i = 1; i <= 10; i++) {
                features.push(parseFloat(formData.get(`x${i}`)));
            }

            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: features })
            });

            const data = await response.json();
            document.getElementById('result').style.display = 'block';
            document.getElementById('prob').innerText = (data.probability * 100).toFixed(2) + '%';
            document.getElementById('class').innerText = data.class === 1 ? 'Positive (1)' : 'Negative (0)';
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([data["features"]], dtype=np.float32)
        
        # Perform inference
        prediction = float(model.predict(features, verbose=0)[0][0])
        predicted_class = 1 if prediction >= 0.5 else 0
        
        return jsonify({
            "status": "success",
            "probability": prediction,
            "class": predicted_class
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
