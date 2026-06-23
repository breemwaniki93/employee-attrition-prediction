import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏢 Employee Attrition Predictor")

st.write("Fill in the employee details below.")

age = st.number_input("Age", min_value=18, max_value=65)

department = st.number_input("Department Code")

salary = st.number_input("Salary")

years_at_company = st.number_input("Years At Company")

overtime_hours = st.number_input("Overtime Hours")

work_satisfaction = st.number_input(
    "Work Satisfaction (1-10)",
    min_value=1,
    max_value=10
)

distance_from_home = st.number_input("Distance From Home")

promotion_last_2_years = st.number_input(
    "Promotion Last 2 Years (0 or 1)",
    min_value=0,
    max_value=1
)

if st.button("Predict Attrition"):

    employee = np.array([[
        age,
        department,
        salary,
        years_at_company,
        overtime_hours,
        work_satisfaction,
        distance_from_home,
        promotion_last_2_years
    ]])

    employee_scaled = scaler.transform(employee)

    prediction = model.predict(employee_scaled)

    if prediction[0] == 1:

        st.error("⚠️ Employee is likely to leave the company.")

    else:

        st.success("✅ Employee is likely to stay with the company.")