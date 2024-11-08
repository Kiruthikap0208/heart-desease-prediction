import streamlit as st
import pandas as pd
import joblib
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def model_selection_page():
    st.title("Model Selection")
    st.write("Select a machine learning model and perform model evaluation.")

    # File upload section
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        # Load and display the dataset
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Overview")
        st.dataframe(data.head())

        # Select target and features
        target = 'target'  # Assuming 'target' is the column for heart disease presence
        X = data.drop(columns=[target])
        y = data[target]

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Model choice radio button
        model_choice = st.radio(
            "Choose a Machine Learning Model",
            ("Logistic Regression", "Support Vector Machine", "Random Forest", "K-Nearest Neighbors")
        )

        # Dictionary of models
        models = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Machine": SVC(),
            "Random Forest": RandomForestClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier()
        }

        # Model selection section
        if model_choice:
            model = models[model_choice]
            model.fit(X_train, y_train)
            st.write(f"### {model_choice} Model Trained")

            # Predictions and evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("**Accuracy:**", accuracy)
            st.write("**Classification Report:**")
            st.text(classification_report(y_test, y_pred))
            st.write("**Confusion Matrix:**")
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            st.pyplot(plt)

            joblib.dump(model, 'heart_disease_best_model.pkl')  # Save the model
            joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
            st.write("Model and scaler saved successfully!")

        # Comparative Analysis Button
        if st.button("Comparative Analysis"):
            st.write("### Comparative Analysis of All Models")
            results = {}

            # Train and evaluate each model
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = math.sqrt(mse)
                cm = confusion_matrix(y_test, y_pred)

                results[name] = {
                    "accuracy": accuracy,
                    "mae": mae,
                    "mse": mse,
                    "rmse": rmse,
                    "confusion_matrix": cm
                }

            # Display results
            metrics_df = pd.DataFrame({
                "Model": results.keys(),
                "Accuracy": [res["accuracy"] for res in results.values()],
                "MAE": [res["mae"] for res in results.values()],
                "MSE": [res["mse"] for res in results.values()],
                "RMSE": [res["rmse"] for res in results.values()]
            })

            # Plot bar charts
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            metrics = ["Accuracy", "MAE", "MSE", "RMSE"]

            for ax, metric in zip(axes.flat, metrics):
                sns.barplot(x="Model", y=metric, data=metrics_df, ax=ax)
                ax.set_title(f"{metric} Comparison")
                ax.bar_label(ax.containers[0], fmt="%.2f")

            plt.tight_layout()
            st.pyplot(fig)

            # Display the best-performing model based on accuracy
            best_model = metrics_df.loc[metrics_df["Accuracy"].idxmax(), "Model"]
            st.write(f"### Conclusion")
            st.write(f"The model that performed the best is **{best_model}**, with an accuracy of **{metrics_df['Accuracy'].max():.2f}**.")




    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    model_selection_page()
