import os
import json
import joblib
from flask import Flask, request

MODEL_ROOT = "model"
main_root = os.getcwd()

# Load the model & all the dependencies
model = joblib.load("log_model.joblib")
scaler = joblib.load("\model\StandardScaler.joblib")
le_gender = joblib.load("\model\LabelEncodergender.joblib")
le_InternetService = joblib.load("\model\LabelEncoderInternetService.joblib")
le_PaperlessBilling = joblib.load("\model\LabelEncoderPaperlessBilling.joblib")
le_Partner = joblib.load("\model\LabelEncoderPartner.joblib")
le_PhoneService = joblib.load("\model\LabelEncoderPhoneService.joblib")
le_SeniorCitizen = joblib.load("\model\LabelEncoderSeniorCitizen.joblib")
le_StreamingTV = joblib.load("\model\LabelEncoderStreamingTV.joblib")

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello, World!"


def preprocessing(data):
    new_data = []
    numeric_data = []
    
    new_data.append(le_gender.transform([data["gender"]])[0])
    new_data.append(le_SeniorCitizen.transform([data["SeniorCitizen"]])[0])
    new_data.append(le_Partner.transform([data["Partner"]])[0])
    new_data.append(le_StreamingTV.transform([data["StreamingTV"]])[0])
    new_data.append(le_PhoneService.transform([data["PhoneService"]])[0])
    new_data.append(le_InternetService.transform([data["InternetService"]])[0])
    new_data.append(le_PaperlessBilling.transform([data["PaperlessBilling"]])[0])
    
    numeric_data.append(data["MonthlyCharges"])
    numeric_data.append(data["TotalCharges"])
    numeric_data.append(data["tenure"])
    new_numeric_data = scaler.transform([numeric_data])[0]
    new_data.append(new_numeric_data[0])
    new_data.append(new_numeric_data[1])
    new_data.append(new_numeric_data[2])
    return new_data


@app.route("/predict", methods=["POST"])
def predict():
    request_json = request.json
    raw_data = request_json.get("data")
    data_preprocessing = preprocessing(raw_data)

    prediction = model.predict([data_preprocessing])
    prediction_string = [str(d) for d in prediction]

    response_json = {
        "raw_data": raw_data,
        "prediction": list(prediction_string)
    }

    return json.dumps(response_json)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
