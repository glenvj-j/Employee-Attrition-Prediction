import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="HR Prediction Tools for Attrition",
    page_icon="📊",
    layout="wide"
)

st.sidebar.success("Click Load Data to Start")
st.image("../image/cover.jpg")

st.markdown(
    """
    <h1 style='text-align: center;'>🔍 HR Attrition Prediction Tool</h1>
    
    <p style='text-align: center;'> Predict employee attrition risk using data-driven insights powered by Logistic Regression.</p>
    
    """,
    unsafe_allow_html=True
)

st.info("👈 Click Prediction from the left panel to run predictions.")

col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    st.markdown("#### 📉 Employee Attrition Analysis")
    st.markdown("---")
    st.write("This tool predicts which employees are at risk of leaving based on historical HR data.")

with col2:
    st.markdown("#### ⚙️ Powered by Logistic Regression")
    st.markdown("---")
    st.write(
        "Using a trained Logistic Regression model, the tool estimates the likelihood of employee attrition. "
        "The model was trained on **1,400 employee records** from the [HR Analytics Dataset](https://www.kaggle.com/datasets/anshika2301/hr-analytics-dataset/data) on Kaggle, "
        "capturing key factors influencing employee retention."
    )
with col3:
    st.markdown("#### 🚀 Data-Driven HR Decisions")
    st.markdown("---")
    st.write("Enhance employee retention strategies by identifying high-risk employees and taking proactive measures.")

st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #7F8C8D;">
    <p>Disclaimer: This tool provides probability-based predictions and should be used alongside HR expertise.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('''
    For details on how the model works, visit: [Click Here](https://github.com/glenvj-j/Employee-Attrition-Prediction)
    
    Created by: Glen Valencius
''')
