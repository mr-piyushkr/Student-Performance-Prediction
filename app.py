import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------- HEADER ----------------
st.title("🎓 Student Performance Prediction")
st.write(
    "This app predicts student performance based on academic inputs. "
    "It is built using a machine learning model."
)

st.divider()

# ---------------- INPUTS ----------------
st.subheader("📥 Enter Student Details")

study_time = st.slider("Study Time (hours/day)", 1, 10, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous_score = st.slider("Previous Exam Score", 0, 100, 60)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Performance"):
    # Dummy logic (real model later)
    score = (study_time * 10 + attendance * 0.4 + previous_score * 0.6) / 2

    if score >= 70:
        st.success("✅ Predicted Performance: Good")
    elif score >= 40:
        st.warning("⚠️ Predicted Performance: Average")
    else:
        st.error("❌ Predicted Performance: Needs Improvement")

    st.info("This is a demo prediction. Model integration will be added next.")
