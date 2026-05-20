from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and encoders safely
model = pickle.load(open("model.pkl", "rb"))
education_encoder = pickle.load(open("education_encoder.pkl", "rb"))
job_encoder = pickle.load(open("job_encoder.pkl", "rb"))
salary_encoder = pickle.load(open("salary_encoder.pkl", "rb"))

@app.route('/')
def home():
    return "Employee Salary Prediction API is Running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        experience = data['experience']
        education = data['education']
        job_role = data['job_role']
        hours_per_week = data['hours_per_week']

        # Encode categorical values
        education_encoded = education_encoder.transform([education])[0]
        job_encoded = job_encoder.transform([job_role])[0]

        # Prepare input
        features = np.array([[experience, education_encoded, job_encoded, hours_per_week]])

        # Predict
        prediction = model.predict(features)

        # Convert back to label
        result = salary_encoder.inverse_transform(prediction)

        return jsonify({
            "predicted_salary_category": result[0]
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

# IMPORTANT for Railway
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)