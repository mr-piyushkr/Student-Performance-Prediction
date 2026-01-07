import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer 
warnings.filterwarnings('ignore')

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
# LOAD DATA AND MODELS
# --------------------------------------------------
@st.cache_data
def load_data_and_models():
    try:
        preprocessor = DataPreprocessor('student_performance.csv')
        X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_data()
        trainer = ModelTrainer(X_train, X_test, y_train, y_test)
        results = trainer.train_and_evaluate()
        
        feature_names = [
            'Previous Sem CGPA', 'Attendance Percentage', 'Extracurricular Activities', 
            'Study Hours per Week', 'Backlogs Previous Sem', 'Internship Experience'
        ]
        
        return trainer, results, scaler, feature_names, preprocessor.data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

trainer, results, scaler, feature_names, raw_data = load_data_and_models()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("🎓 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Home", "📊 Data Analysis", "🤖 Model Performance", "🔮 Prediction", "📈 Insights"]
)

if results:
    best_model = max(results.keys(), key=lambda x: results[x]['R2 Score'])
    best_score = results[best_model]['R2 Score']
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Stats")
    st.sidebar.metric("Best Model", best_model)
    st.sidebar.metric("Best R² Score", f"{best_score:.3f}")
    st.sidebar.metric("Total Features", "6")
    if raw_data is not None:
        st.sidebar.metric("Dataset Size", len(raw_data))

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "🏠 Home":
    st.title("🎓 Student Performance Prediction System")
    st.markdown("""
    Welcome to the **Student Performance Prediction System**! This application uses machine learning 
    to predict student academic performance based on various factors.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📊 Data Analysis**
        
        Explore the dataset with interactive visualizations and statistical insights.
        """)
    
    with col2:
        st.success("""
        **🤖 ML Models**
        
        Compare performance of 4 different machine learning algorithms.
        """)
    
    with col3:
        st.warning("""
        **🔮 Predictions**
        
        Get real-time CGPA predictions with personalized recommendations.
        """)
    
    st.markdown("---")
    
    if results:
        st.subheader("📊 Model Performance Overview")
        
        # Create performance comparison chart
        models = list(results.keys())
        r2_scores = [results[model]['R2 Score'] for model in models]
        
        fig = px.bar(
            x=models, y=r2_scores,
            title="Model R² Score Comparison",
            labels={'x': 'Models', 'y': 'R² Score'},
            color=r2_scores,
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### 🎯 Key Features
    
    - **Multi-Model Comparison**: Random Forest, SVM, Linear & Ridge Regression
    - **Interactive Visualizations**: Plotly charts for better insights
    - **Real-time Predictions**: Instant CGPA forecasting
    - **Feature Analysis**: Understand what drives academic success
    - **Data Insights**: Comprehensive dataset exploration
    """)

# --------------------------------------------------
# DATA ANALYSIS PAGE
# --------------------------------------------------
elif page == "📊 Data Analysis":
    st.title("📊 Dataset Analysis")
    
    if raw_data is not None:
        st.subheader("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(raw_data))
        with col2:
            st.metric("Features", len(raw_data.columns)-1)
        with col3:
            st.metric("Avg CGPA", f"{raw_data['next_sem_cgpa'].mean():.2f}")
        with col4:
            st.metric("Max CGPA", f"{raw_data['next_sem_cgpa'].max():.2f}")
        
        st.markdown("---")
        
        # Data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(raw_data.head(10), use_container_width=True)
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(raw_data.describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("📊 Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CGPA Distribution
            fig = px.histogram(
                raw_data, x='next_sem_cgpa', nbins=20,
                title="Distribution of Next Semester CGPA",
                labels={'next_sem_cgpa': 'Next Semester CGPA', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Attendance vs Performance
            fig = px.scatter(
                raw_data, x='attendance_percentage', y='next_sem_cgpa',
                title="Attendance vs Performance",
                labels={'attendance_percentage': 'Attendance %', 'next_sem_cgpa': 'Next Sem CGPA'},
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Heatmap
        st.subheader("🔥 Feature Correlation Matrix")
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        corr_matrix = raw_data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix, text_auto=True, aspect="auto",
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data available for analysis.")

# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance Analysis")
    
    if results:
        st.subheader("📊 Performance Metrics Comparison")
        
        # Performance metrics table
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df = results_df.round(4)
        st.dataframe(results_df, use_container_width=True)
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            models = list(results.keys())
            r2_scores = [results[model]['R2 Score'] for model in models]
            
            fig = go.Figure(data=[
                go.Bar(x=models, y=r2_scores, name='R² Score',
                      marker_color='lightblue')
            ])
            fig.update_layout(title='Model R² Scores', yaxis_title='R² Score')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # MAE comparison
            mae_scores = [results[model]['MAE'] for model in models]
            
            fig = go.Figure(data=[
                go.Bar(x=models, y=mae_scores, name='MAE',
                      marker_color='lightcoral')
            ])
            fig.update_layout(title='Model Mean Absolute Error', yaxis_title='MAE')
            st.plotly_chart(fig, use_container_width=True)
        
        # Best model highlight
        best_model = max(results.keys(), key=lambda x: results[x]['R2 Score'])
        st.success(f"🏆 **Best Performing Model**: {best_model} with R² Score of {results[best_model]['R2 Score']:.4f}")
        
        # Model details
        st.subheader("🔍 Model Details")
        selected_model = st.selectbox("Select a model to view details:", list(results.keys()))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{results[selected_model]['R2 Score']:.4f}")
        with col2:
            st.metric("MAE", f"{results[selected_model]['MAE']:.4f}")
        with col3:
            st.metric("MSE", f"{results[selected_model]['MSE']:.4f}")
    else:
        st.error("No model results available.")

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------
elif page == "🔮 Prediction":
    st.title("🔮 CGPA Prediction")
    st.markdown("Enter student details below to get a CGPA prediction using our trained models.")
    
    if trainer and scaler:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Academic Information")
            prev_cgpa = st.slider("Previous Semester CGPA", 0.0, 10.0, 7.5, 0.1)
            attendance = st.slider("Attendance Percentage", 0, 100, 85)
            backlogs = st.selectbox("Previous Semester Backlogs", [0, 1, 2, 3, 4, 5])
        
        with col2:
            st.subheader("🎯 Personal Factors")
            study_hours = st.slider("Study Hours per Week", 0, 60, 25)
            extracurricular = st.selectbox("Extracurricular Activities", [0, 1], 
                                         format_func=lambda x: "Yes" if x else "No")
            internship = st.selectbox("Internship Experience", [0, 1], 
                                    format_func=lambda x: "Yes" if x else "No")
        
        st.markdown("---")
        
        if st.button("🚀 Predict CGPA", type="primary"):
            # Prepare input data
            input_data = np.array([[prev_cgpa, attendance, extracurricular, study_hours, backlogs, internship]])
            input_scaled = scaler.transform(input_data)
            
            # Get predictions from all models
            predictions = {}
            for name, model in trainer.models.items():
                pred = model.predict(input_scaled)[0]
                predictions[name] = max(0, min(10, pred))
            
            # Calculate ensemble prediction
            ensemble_pred = np.mean(list(predictions.values()))
            
            st.subheader("📊 Prediction Results")
            
            # Main prediction
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if ensemble_pred >= 8.5:
                    st.success(f"🌟 **Predicted CGPA: {ensemble_pred:.2f}** - Excellent Performance Expected!")
                elif ensemble_pred >= 7.0:
                    st.info(f"✅ **Predicted CGPA: {ensemble_pred:.2f}** - Good Performance Expected!")
                elif ensemble_pred >= 6.0:
                    st.warning(f"⚠️ **Predicted CGPA: {ensemble_pred:.2f}** - Average Performance Expected")
                else:
                    st.error(f"❌ **Predicted CGPA: {ensemble_pred:.2f}** - Needs Improvement")
            
            with col2:
                # Progress bar
                progress = min(ensemble_pred / 10, 1.0)
                st.metric("Prediction Confidence", f"{progress*100:.1f}%")
                st.progress(progress)
            
            # Individual model predictions
            st.subheader("🤖 Individual Model Predictions")
            pred_cols = st.columns(len(predictions))
            for i, (model, pred) in enumerate(predictions.items()):
                with pred_cols[i]:
                    st.metric(model, f"{pred:.2f}")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            recommendations = []
            
            if attendance < 75:
                recommendations.append("📅 **Improve Attendance**: Aim for 85%+ attendance")
            if study_hours < 20:
                recommendations.append("📚 **Increase Study Time**: Consider 25+ hours per week")
            if not extracurricular:
                recommendations.append("🎯 **Join Activities**: Extracurricular activities help")
            if not internship:
                recommendations.append("💼 **Gain Experience**: Internships provide valuable knowledge")
            if backlogs > 0:
                recommendations.append("🎯 **Clear Backlogs**: Focus on clearing pending subjects")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"• {rec}")
            else:
                st.success("🎉 Great job! You're on track for excellent performance!")
    else:
        st.error("Models not available for prediction.")

# --------------------------------------------------
# INSIGHTS PAGE
# --------------------------------------------------
elif page == "📈 Insights":
    st.title("📈 Feature Importance & Insights")
    
    if trainer and feature_names:
        # Feature importance from Random Forest
        rf_model = trainer.models['Random Forest']
        importances = rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        st.subheader("🎯 Feature Importance Analysis")
        
        # Feature importance chart
        fig = px.bar(
            importance_df, x='Importance', y='Feature',
            orientation='h', title="Feature Importance in CGPA Prediction",
            color='Importance', color_continuous_scale='viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Key insights
        st.subheader("🔍 Key Insights")
        
        top_feature = importance_df.iloc[-1]['Feature']
        top_importance = importance_df.iloc[-1]['Importance']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **🎯 Most Important Factor**
            
            {top_feature} is the most influential factor with {top_importance:.1%} importance.
            """)
        
        with col2:
            st.success("""
            **📚 Academic Success Tips**
            
            Focus on maintaining good attendance, consistent study habits, and previous academic performance.
            """)
        
        # Detailed insights
        st.subheader("📊 Detailed Analysis")
        
        insights = [
            f"🎯 **{top_feature}** is the most important factor ({top_importance:.1%} importance)",
            "📚 **Academic factors** (Previous CGPA, Study Hours) are crucial predictors",
            "📅 **Attendance** plays a significant role in academic success",
            "🎭 **Extracurricular activities** contribute to overall development",
            "💼 **Internship experience** provides practical knowledge"
        ]
        
        for insight in insights:
            st.markdown(f"• {insight}")
        
        # Feature importance table
        st.subheader("📋 Feature Importance Table")
        importance_df_sorted = importance_df.sort_values('Importance', ascending=False)
        st.dataframe(importance_df_sorted, use_container_width=True)
    else:
        st.error("Feature importance data not available.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with ❤️ using Streamlit | Student Performance Prediction System</div>",
    unsafe_allow_html=True
)