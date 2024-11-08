import streamlit as st

def about_page():
    st.title("Heart Disease Prediction Model")
    st.markdown("""
    This web application uses machine learning models to predict the likelihood of heart disease based on user input.
    
    ### Features:
    - **Data Analysis**: Explore and analyze the uploaded dataset to see trends and relationships.
    - **Model Selection**: Train and evaluate different models to find the best-performing model.
    - **Check**: Enter patient information to predict the likelihood of heart disease based on the trained model.
    """)

if __name__ == "__main__":
    about_page()
