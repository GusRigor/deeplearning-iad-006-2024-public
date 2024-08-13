from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

with open('modelo.pkl', 'rb') as file:
    modelo = pickle.load(file)

with open('modelo-xgboost.pkl', 'rb') as file:
    modelo_xgboost = pickle.load(file)

@app.route('/predict', methods=['POST'])
@app.route('/random_forest/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print(data)
    prediction = modelo.predict([data['input']])
    return jsonify({'prediction': prediction.tolist()})


@app.route('/xgboost/predict', methods=['POST'])
def predict_xgboost():
    data = request.get_json(force=True)
    print(data)
    prediction = modelo_xgboost.predict([data['input']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
