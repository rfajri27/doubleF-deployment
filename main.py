import json
import joblib
from flask import Flask, request

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello, World!"


# Load the model
model = joblib.load("model\log_model.joblib")
scaler = joblib.load("model\StandardScaler.joblib")
label_encoder = joblib.load("model\LabelEncoder.joblib")

def preprocessing(data):
    data[:7] = label_encoder.transform(data[:7])
    data[-3:] = scaler.transform(data[-3:])
    
    return data


@app.route("/predict", methods=["POST"])
def predict():
    request_json = request.json
    data = preprocessing(request_json.get("data"))

    prediction = model.predict(data)
    prediction_string = [str(d) for d in prediction]

    response_json = {
        "raw_data": request_json.get("data"),
        "data": data,
        "prediction": list(prediction_string)
    }

    return json.dumps(response_json)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
