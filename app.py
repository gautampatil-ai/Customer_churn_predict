import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load extracted model weights
weights_path = os.path.join(os.path.dirname(__file__), "weights.json")

def load_weights():
    if os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            data = json.load(f)
            return {k: np.array(v, dtype=np.float32) for k, v in data.items()}
    return None

WEIGHTS = load_weights()

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_forward(inputs):
    """
    Recreates forward pass for: Input(10) -> Dense(8, relu) -> Dense(7, relu) -> Dense(1, sigmoid)
    """
    # Keys matching internal Keras HDF5 structure
    w0 = WEIGHTS["vars/0"]  # Dense 1 Kernel (10, 8)
    b0 = WEIGHTS["vars/1"]  # Dense 1 Bias (8,)
    w1 = WEIGHTS["vars/2"]  # Dense 2 Kernel (8, 7)
    b1 = WEIGHTS["vars/3"]  # Dense 2 Bias (7,)
    w2 = WEIGHTS["vars/4"]  # Dense 3 Kernel (7, 1)
    b2 = WEIGHTS["vars/5"]  # Dense 3 Bias (1,)

    # Forward Pass Computation
    layer1 = relu(np.dot(inputs, w0) + b0)
    layer2 = relu(np.dot(layer1, w1) + b1)
    output = sigmoid(np.dot(layer2, w2) + b2)
    
    return float(output[0][0])

# Modern UI Dashboard Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Predictor</title>
    <style>
        body { background-color: #0e1117; color: #ffffff; font-family: sans-serif; padding: 2rem; }
        .container { max-width: 800px; margin: 0 auto; background: #1e222d; padding: 2rem; border-radius: 12px; border: 1px solid #2e364f; }
        h1 { color: #4CAF50; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
        .input-group { display: flex; flex-direction: column; }
        label { font-size: 0.8rem; color: #a0a0a0; margin-bottom: 0.3rem; }
        input { background: #0e1117; border: 1px solid #2e364f; color: white; padding: 0.5rem; border-radius: 6px; }
        button { width: 100%; background: #4CAF50; color: white; border: none; padding: 0.8rem; font-size: 1rem; font-weight: bold; border-radius: 6px; cursor: pointer; }
        .result-box { margin-top: 1.5rem; padding: 1rem; background: #0e1117; border-radius: 8px; display: none; }
        .metric { font-size: 1.5rem; font-weight: bold; color: #4CAF50; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Customer Churn Neural Predictor</h1>
        <p>Enter 10 feature values for real-time model inference:</p>
        <form id="predictForm">
            <div class="grid">
                {% for i in range(1, 11) %}
                <div class="input-group">
                    <label>Feature X_{{ i }}</label>
                    <input type="number" step="0.01" name="x{{ i }}" value="0.00" required>
                </div>
                {% endfor %}
            </div>
            <button type="submit">Predict Churn Risk</button>
        </form>
        
        <div id="result" class="result-box">
            <h3>Inference Results:</h3>
            <p>Predicted Probability: <span id="prob" class="metric">--</span></p>
            <p>Classification Output: <span id="class" class="metric">--</span></p>
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
            document.getElementById('class').innerText = data.class === 1 ? 'High Risk (1)' : 'Low Risk (0)';
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
        
        prob = predict_forward(features)
        predicted_class = 1 if prob >= 0.5 else 0
        
        return jsonify({
            "status": "success",
            "probability": prob,
            "class": predicted_class
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
