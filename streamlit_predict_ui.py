import streamlit as st
import requests

st.title("Salary Prediction")

# Collect user input
user_input = {
    "id": st.number_input("ID", value=101),
    "age": st.number_input("Age", value=35),
    "years_experience": st.number_input("Years Experience", value=10),
    "department": st.text_input("Department", value="Sales"),
    "education": st.text_input("Education", value="Bachelors"),
    "education_level": st.text_input("Education Level", value="Undergraduate"),
    "performance_score": st.number_input("Performance Score", value=4.2),
    "remote_work_pct": st.number_input("Remote Work %", value=20),
    "bonus": st.number_input("Bonus", value=5000),
    "hire_date": st.text_input("Hire Date", value="2015-06-01"),
    "certifications": st.number_input("Certifications", value=2)
}

if st.button("Predict Salary"):
    payload = {"records": [user_input]}
    #response = requests.post("http://127.0.0.1:9000/predict", json=payload)
    #response = requests.post("http://127.0.0.1:9000/predict", json=payload) 
    response = requests.post("http://54.234.127.110:9000/predict", json=payload)
    if response.status_code == 200:
        st.success(f"Predicted Salary: {response.json()['predictions'][0]:,.2f}")
    else:
        st.error(f"Error: {response.text}")