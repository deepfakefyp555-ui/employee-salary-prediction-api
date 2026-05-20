import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
data = pd.read_csv("salary_data.csv")

# Convert text into numbers
le_education = LabelEncoder()
le_job = LabelEncoder()
le_salary = LabelEncoder()

data['education'] = le_education.fit_transform(data['education'])
data['job_role'] = le_job.fit_transform(data['job_role'])
data['salary_category'] = le_salary.fit_transform(data['salary_category'])

# Inputs and output
X = data[['experience', 'education', 'job_role', 'hours_per_week']]
y = data['salary_category']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model and encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_education, open("education_encoder.pkl", "wb"))
pickle.dump(le_job, open("job_encoder.pkl", "wb"))
pickle.dump(le_salary, open("salary_encoder.pkl", "wb"))

print("Model trained successfully!")