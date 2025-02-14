import streamlit as st
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
# Load trained models
model1 = joblib.load("logisticregression_heart_disease_model.pkl")
model2 = joblib.load("knn_heart_disease_model.pkl")
model3 = joblib.load("svc_heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction System")
st.write("Enter the required details below to check the heart disease prediction.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=0)
chol = st.number_input("Cholesterol Level", min_value=0)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Convert categorical values to numerical
sex = 1 if sex == "Male" else 0

# Make predictions
if st.button("Check Prediction"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    reshape = input_data.reshape(1,-1)
    
    prediction1 = model1.predict(input_data)[0]
    prediction2 = model2.predict(input_data)[0]
    prediction3 = model3.predict(input_data)[0]
    
    st.subheader("Prediction Results:")
    st.write(f"Prediction 1: {'Heart Disease Detected' if prediction1 == 1 else 'No Heart Disease'}")
    st.write(f"Prediction 2: {'Heart Disease Detected' if prediction2 == 1 else 'No Heart Disease'}")
    st.write(f"Prediction 3: {'Heart Disease Detected' if prediction3 == 1 else 'No Heart Disease'}")
