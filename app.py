import streamlit as st
import numpy as np
import pandas as pd

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS (PREMIUM LOOK)
# --------------------------------------------------
st.markdown("""
<style>
/* Global */
body {
    background-color: #f8fafc;
}
.block-container {
    padding-top: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #020617);
}
[data-testid="stSidebar"] * {
    color: white;
}

/* Cards */
.card {
    background: white;
    padding: 28px;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Titles */
.main-title {
    font-size: 46px;
    font-weight: 800;
    color: #020617;
}
.sub-title {
    font-size: 18px;
    color: #475569;
}

/* Metrics */
.metric-box {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
}
.metric-box h2 {
    margin: 0;
    font-size: 34px;
}
.metric-box p {
    margin: 0;
    font-size: 14px;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR NAV
# --------------------------------------------------
st.sidebar.markdown("## 🎯 Navigation")
page = st.sidebar.radio(
    "",
    ["🏠 Overview", "📊 Insights", "🔮 Prediction"]
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="main-title">Student Performance Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A clean, production-ready machine learning demo built with Python & Streamlit</div>',
    unsafe_allow_html=True
)
st.write("")

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
if page == "🏠 Overview":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
### 📘 About the Project
This application demonstrates how student academic performance can be predicted
using data-driven techniques.

The focus of this project is **clarity, structure, and real-world usability** —
not just model accuracy.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="metric-box"><h2>ML</h2><p>Prediction System</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-box"><h2>Python</h2><p>Core Language</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric-box"><h2>Streamlit</h2><p>Interactive UI</p></div>', unsafe_allow_html=True)

# --------------------------------------------------
# INSIGHTS (STATIC + FAST)
# --------------------------------------------------
elif page == "📊 Insights":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Performance Insights")

    data = pd.DataFrame({
        "Category": ["Excellent", "Good", "Average", "Needs Improvement"],
        "Percentage": [30, 40, 20, 10]
    })

    st.bar_chart(data.set_index("Category"))
    st.caption("Sample aggregated distribution for demonstration purposes.")

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# PREDICTION (FAST & CLEAN)
# --------------------------------------------------
elif page == "🔮 Prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔮 Predict Student Performance")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider("Study Hours per Day", 1, 10, 5)
        attendance = st.slider("Attendance (%)", 0, 100, 75)

    with col2:
        previous_score = st.slider("Previous Exam Score", 0, 100, 60)
        sleep_hours = st.slider("Sleep Hours", 3, 10, 7)

    if st.button("🚀 Predict Performance"):
        score = (
            study_hours * 0.3 +
            attendance * 0.3 +
            previous_score * 0.3 +
            sleep_hours * 0.1
        )

        if score >= 75:
            st.success("🌟 Predicted Performance: Excellent")
        elif score >= 60:
            st.info("✅ Predicted Performance: Good")
        elif score >= 40:
            st.warning("⚠️ Predicted Performance: Average")
        else:
            st.error("❌ Predicted Performance: Needs Improvement")

        st.caption("Prediction logic is simplified for demonstration.")

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    "<p style='text-align:center; color:#64748b;'>Built by Piyush Kumar • Machine Learning & Software Development</p>",
    unsafe_allow_html=True
)
