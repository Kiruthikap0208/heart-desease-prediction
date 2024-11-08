import streamlit as st
from about_page import about_page
from data_analysis_page import data_analysis_page
from model_selection_page import model_selection_page
from check_page import check_page

# Set Streamlit page configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Sidebar title and navigation with radio buttons
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("About", "Data Analysis", "Model Selection", "Check")
)

# Render the selected page based on the radio selection
if page == "About":
    about_page()
elif page == "Data Analysis":
    data_analysis_page()
elif page == "Model Selection":
    model_selection_page()
elif page == "Check":
    check_page()
