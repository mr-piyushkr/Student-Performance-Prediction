import streamlit as st
import numpy as np
import pickle
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

# Check if model files exist
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    st.error("Model files not found. Please train the model first.")
    st.stop()

model, scaler = load_model()

# ---------------- UI ----------------
st.title("🎓 Student Performance Prediction")
st.write(
    "This application predicts **Next Semester CGPA** based on academic "
    "and activity-related factors using a trained Machine Learning model."
)

st.divider()

st.subheader("📥 Enter Student Details")

previous_cgpa = st.slider(
    "Previous Semester CGPA", 0.0, 10.0, 7.0, step=0.1
)
attendance = st.slider(
    "Attendance Percentage (%)", 0, 100, 75
)
extracurricular = st.slider(
    "Extracurricular Activities (hours/week)", 0, 10, 2
)
study_hours = st.slider(
    "Study Hours per Week", 0, 60, 20
)
backlogs = st.selectbox(
    "Number of Backlogs in Previous Semester", [0, 1, 2, 3]
)
internship = st.selectbox(
    "Internship Experience", ["No", "Yes"]
)

# Convert internship to numeric
internship_val = 1 if internship == "Yes" else 0

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict CGPA"):
    input_data = np.array([[
        previous_cgpa,
        attendance,
        extracurricular,
        study_hours,
        backlogs,
        internship_val
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.subheader("📊 Prediction Result")
    st.metric(
        label="Predicted Next Semester CGPA",
        value=f"{prediction:.2f}"
    )

    if prediction >= 8:
        st.success("🌟 Excellent performance expected")
    elif prediction >= 6:
        st.info("✅ Good performance expected")
    elif prediction >= 4:
        st.warning("⚠️ Average performance expected")
    else:
        st.error("❌ Performance needs improvement")

    st.progress(min(int(prediction * 10), 100))

st.divider()

st.caption(
    "⚙️ Model trained using multiple regression algorithms and selected "
    "based on performance metrics. This tool is for academic demonstration purposes."
)
