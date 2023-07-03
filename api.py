from flask import Flask, request, jsonify
from flask_cors import CORS
from detoxify import Detoxify
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)
predictor = Detoxify('multilingual')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentence = data['sentence']
    results = predictor.predict(sentence)
    results = {k: v.item() if isinstance(v, np.float32) else v for k, v in results.items()}
    return jsonify(results)

if __name__ == '__main__':
    app.run()
