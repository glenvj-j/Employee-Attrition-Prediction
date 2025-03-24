# Employee Attrition Prediction

> [!NOTE]
> Streamlit : [Employee Attrition Prediction](https://employee-attrition-prediction-new.streamlit.app/)

> Dataset from [HR Analytics Dataset](https://www.kaggle.com/datasets/anshika2301/hr-analytics-dataset/data)

## Overview

This project focuses on predicting employee attrition using machine learning. The dataset is sourced from Kaggle, containing 1,400+ employee records. A **Logistic Regression model** is trained to estimate the likelihood of attrition.

## Data Preprocessing

- **Duplicate Removal:** Identical records were dropped.
- **Handling Missing Values:** Missing values in *YearsWithCurrManager* were imputed using the median.
- **Feature Engineering:**
  - Converted *Attrition* column into binary (1 = Yes, 0 = No).
  - Dropped non-informative columns (*EmployeeCount, EmployeeNumber, Over18, StandardHours*).
  - Standardized categorical variables such as *BusinessTravel*.

## Model Training

- **Algorithm Used:** Logistic Regression
- **Feature Scaling:** Applied for numerical variables
- **Model Evaluation:** Confusion matrix and SHAP values analyzed feature importance.

## Deployment

The model is integrated into a **Streamlit** web app, allowing users to upload employee data and receive attrition predictions. Employees are categorized into:

- **High Risk** (most likely to leave)
- **Medium Risk** (potential attrition)
- **Low Risk** (least likely to leave)

The tool provides easy data export options for further HR analysis.

## How to Use

1. Click the **Load Data** button on the left to upload a CSV file with employee data.
2. Click **Predict** to generate attrition probabilities.
3. View categorized risk levels and export results.

---

**Author:** Glen Valencius
