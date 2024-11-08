import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def check_page():
    st.title("Check Your Heart Disease Risk")
    
    # Load the trained model and scaler
    try:
        model = joblib.load('heart_disease_best_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        st.error("Error loading model or scaler. Please ensure the model and scaler files exist.")
        return

    st.write("""
    ### Enter your details to check if you are prone to heart disease.
    Fill in the form below with your details.
    """)

    # Input fields for user data
    age = st.number_input("Age", min_value=0, max_value=120, value=30, help="Your age in years.")
    sex = st.radio("Sex", ('Male', 'Female'), help="Select 'Male' or 'Female'.")
    chest_pain_type = st.selectbox("Chest Pain Type", 
                                   ('Type 0', 'Type 1', 'Type 2', 'Type 3'),
                                   help="Type of chest pain: \n- Type 0: No pain\n- Type 1: Typical angina\n- Type 2: Atypical angina\n- Type 3: Non-anginal pain")
    resting_blood_pressure = st.number_input("Resting Blood Pressure (trestbps)", 
                                             min_value=0, max_value=300, value=120, 
                                             help="Resting blood pressure in mm Hg on admission.")
    cholesterol = st.number_input("Cholesterol (chol)", 
                                  min_value=0, max_value=600, value=200, 
                                  help="Serum cholesterol in mg/dl.")
    fasting_blood_sugar = st.radio("Fasting Blood Sugar (fbs)", 
                                   ('< 120 mg/dl', '> 120 mg/dl'),
                                   help="Fasting blood sugar: '> 120 mg/dl' indicates higher risk.")
    resting_electrocardiographic = st.selectbox("Resting Electrocardiographic Results (restecg)", 
                                                ('Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'),
                                                help="Results of resting electrocardiographic test:\n- Normal\n- ST-T wave abnormality\n- Possible or definite left ventricular hypertrophy")
    max_heart_rate_achieved = st.number_input("Max Heart Rate Achieved (thalach)", 
                                              min_value=0, max_value=250, value=150, 
                                              help="Maximum heart rate achieved during exercise.")
    exercise_induced_angina = st.radio("Exercise-Induced Angina (exang)", 
                                       ('No', 'Yes'),
                                       help="Whether exercise-induced angina occurred: 'Yes' or 'No'.")
    oldpeak = st.number_input("Oldpeak", 
                              min_value=0.0, max_value=10.0, value=1.0, 
                              help="ST depression induced by exercise relative to rest, indicating abnormality.")
    slope = st.selectbox("Slope of the peak exercise ST segment (slope)", 
                         ('Up', 'Flat', 'Down'),
                         help="Slope of the peak exercise ST segment:\n- Up: Upsloping\n- Flat: Flat\n- Down: Downsloping")
    number_of_vessels_fluro = st.selectbox("Number of vessels colored by fluoroscopy (ca)", 
                                           (0, 1, 2, 3),
                                           help="Number of major vessels (0-3) colored by fluoroscopy.")
    thalassemia = st.selectbox("Thalassemia (thal)", 
                               ('Normal', 'Fixed Defect', 'Reversible Defect'),
                               help="Thalassemia test results:\n- Normal\n- Fixed Defect\n- Reversible Defect")

    # Mapping categorical data to numeric
    sex_map = {'Male': 1, 'Female': 0}
    chest_pain_map = {'Type 0': 0, 'Type 1': 1, 'Type 2': 2, 'Type 3': 3}
    fasting_blood_sugar_map = {'< 120 mg/dl': 0, '> 120 mg/dl': 1}
    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    exercise_induced_angina_map = {'No': 0, 'Yes': 1}
    slope_map = {'Up': 0, 'Flat': 1, 'Down': 2}
    thalassemia_map = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}

    # Prepare the user input for prediction
    user_input = np.array([
        age,
        sex_map[sex],
        chest_pain_map[chest_pain_type],
        resting_blood_pressure,
        cholesterol,
        fasting_blood_sugar_map[fasting_blood_sugar],
        restecg_map[resting_electrocardiographic],
        max_heart_rate_achieved,
        exercise_induced_angina_map[exercise_induced_angina],
        oldpeak,
        slope_map[slope],
        number_of_vessels_fluro,
        thalassemia_map[thalassemia]
    ]).reshape(1, -1)

    # When the 'Predict' button is pressed
    if st.button("Predict"):
        # Standardize the input using the loaded scaler
        user_input_scaled = scaler.transform(user_input)

        # Predict the result
        prediction = model.predict(user_input_scaled)

        # Show the prediction result
        if prediction == 1:
            st.write("### Prediction: **You are at risk of heart disease.**")
            st.write("We recommend consulting a healthcare provider for further evaluation.")
        else:
            st.write("### Prediction: **You are not at risk of heart disease.**")
            st.write("Keep up with a healthy lifestyle!")

if __name__ == "__main__":
    check_page()
