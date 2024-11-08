import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_analysis_page():
    st.title("Data Analysis")
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("""The dataset contains **918 rows** and **12 columns**, with the following features:

1. **Age**: The age of the patient (numeric).
2. **Sex**: The gender of the patient ('M' for male, 'F' for female).
3. **Cp**: Type of chest pain experienced by the patient. The values are:
   - **ATA**: Atypical Angina
   - **NAP**: Non-Anginal Pain
   - **ASY**: Asymptomatic
4. **RBP**: Resting blood pressure (in mm Hg).
5. **Chol**: Serum cholesterol level (in mg/dL).
6. **FBS**: Fasting blood sugar (1 if FastingBS > 120 mg/dL, 0 otherwise).
7. **restecg**: Resting electrocardiographic results, with values:
   - **Normal**
   - **ST**: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
   - **LVH**: Left Ventricular Hypertrophy
8. **MaxHR**: Maximum heart rate achieved.
9. **Exan**: Whether exercise-induced angina was present ('Y' for yes, 'N' for no).
10. **Oldpeak**: Depression in the ST segment of an ECG during exercise relative to rest (numeric).
11. **Slope**: The slope of the peak exercise ST segment:
    - **Up**
    - **Flat**
    - **Down**
12. **target**: The target variable, indicating the presence of heart disease (1: heart disease, 0: no heart disease).

- The dataset has both **numeric** and **categorical** features.
""")
        st.write("### Dataset Overview")
        st.dataframe(data.head())

        st.write("**Shape of the dataset:**", data.shape)
        st.write("**Summary statistics:**")
        st.write(data.describe())

        if st.button("Show Data Analysis"):
            st.write("### Data Analysis")
            
        
            # 1. Distribution of patients with or without heart problems (Target)
            st.write("**Distribution of Patients with or without Heart Disease**")
            fig, ax = plt.subplots()
            sns.countplot(x='target', data=data, palette=['#66c2a5', '#fc8d62'])
            ax.set_xticklabels(['No Heart Disease', 'Heart Disease'])
            ax.set_ylabel("Count")
            ax.set_title("Count of Patients with and without Heart Disease")
            st.pyplot(fig)

            # 2. Target versus Sex
            st.write("**Heart Disease Distribution by Sex**")
            fig, ax = plt.subplots()
            sns.countplot(data=data, x='sex', hue='target', palette="Set2")
            ax.set_xticklabels(['Female', 'Male'])
            ax.set_xlabel("Sex")
            ax.set_ylabel("Count")
            ax.set_title("Heart Disease by Sex")
            st.pyplot(fig)

            # 3. Target versus Cholesterol Level
            st.write("**Cholesterol Level Distribution by Heart Disease Status**")
            fig, ax = plt.subplots()
            sns.kdeplot(data=data[data["target"] == 0]["chol"], label="No Heart Disease", shade=True, color="blue")
            sns.kdeplot(data=data[data["target"] == 1]["chol"], label="Heart Disease", shade=True, color="red")
            ax.set_xlabel("Cholesterol Level")
            ax.set_title("Cholesterol Levels for Patients with and without Heart Disease")
            ax.legend(loc="upper right")
            st.pyplot(fig)

            # 4. Target versus Blood Pressure
            st.write("**Blood Pressure (trestbps) Distribution by Heart Disease Status**")
            fig, ax = plt.subplots()
            sns.kdeplot(data=data[data["target"] == 0]["trestbps"], label="No Heart Disease", shade=True, color="blue")
            sns.kdeplot(data=data[data["target"] == 1]["trestbps"], label="Heart Disease", shade=True, color="red")
            ax.set_xlabel("Resting Blood Pressure")
            ax.set_title("Blood Pressure Levels for Patients with and without Heart Disease")
            ax.legend(loc="upper right")
            st.pyplot(fig)

            # 5. Target versus Age
            st.write("**Age Distribution by Heart Disease Status**")
            fig, ax = plt.subplots()
            sns.kdeplot(data=data[data["target"] == 0]["age"], label="No Heart Disease", shade=True, color="blue")
            sns.kdeplot(data=data[data["target"] == 1]["age"], label="Heart Disease", shade=True, color="red")
            ax.set_xlabel("Age")
            ax.set_title("Age Distribution for Patients with and without Heart Disease")
            ax.legend(loc="upper right")
            st.pyplot(fig)

            # 6. Target versus Fasting Blood Sugar (fbs)
            st.write("**Heart Disease Distribution by Fasting Blood Sugar Level**")
            fig, ax = plt.subplots()
            sns.countplot(data=data, x='fbs', hue='target', palette="muted")
            ax.set_xticklabels(['FBS < 120 mg/dl', 'FBS > 120 mg/dl'])
            ax.set_xlabel("Fasting Blood Sugar")
            ax.set_ylabel("Count")
            ax.set_title("Heart Disease by Fasting Blood Sugar Level")
            st.pyplot(fig)

            # 7. Target versus Exercise-Induced Angina (exang)
            st.write("**Heart Disease Distribution by Exercise-Induced Angina**")
            fig, ax = plt.subplots()
            sns.countplot(data=data, x='exang', hue='target', palette="muted")
            ax.set_xticklabels(['No Angina', 'Exercise-Induced Angina'])
            ax.set_xlabel("Exercise-Induced Angina")
            ax.set_ylabel("Count")
            ax.set_title("Heart Disease by Exercise-Induced Angina")
            st.pyplot(fig)

            # 8. Target versus Chest Pain Type (cp)
            st.write("**Heart Disease Distribution by Chest Pain Type**")
            fig, ax = plt.subplots()
            sns.countplot(data=data, x='cp', hue='target', palette="muted")
            ax.set_xticklabels(['Type 0', 'Type 1', 'Type 2', 'Type 3'])
            ax.set_xlabel("Chest Pain Type")
            ax.set_ylabel("Count")
            ax.set_title("Heart Disease by Chest Pain Type")
            st.pyplot(fig)
if __name__ == "__main__":
    data_analysis_page()
