from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load files
model = pickle.load(open("model.pkl", "rb"))
education_encoder = pickle.load(open("education_encoder.pkl", "rb"))
job_encoder = pickle.load(open("job_encoder.pkl", "rb"))
salary_encoder = pickle.load(open("salary_encoder.pkl", "rb"))

@app.route('/')
def home():
    return "Employee Salary Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    experience = data['experience']
    education = data['education']
    job_role = data['job_role']
    hours_per_week = data['hours_per_week']

    # Convert text to numbers
    education_encoded = education_encoder.transform([education])[0]
    job_encoded = job_encoder.transform([job_role])[0]

    # Prepare input
    features = np.array([[experience,
                          education_encoded,
                          job_encoded,
                          hours_per_week]])

    # Prediction
    prediction = model.predict(features)

    # Convert back to text
    result = salary_encoder.inverse_transform(prediction)

    return jsonify({
        "predicted_salary_category": result[0]
    })

if __name__ == '__main__':
    app.run(debug=True)