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

- ## Model Performance Comparison

 ## 📊 Cost Savings Overview  

By implementing the attrition prediction model, the company can **save up to 880,000** by identifying employees at risk of leaving and taking preventive measures.

| **Metric**               | **Base Model** | **Tuned Model** | **Improvement** |
|--------------------------|--------------|--------------|---------------|
| **Cost Reduction**       | 1,560,000    | 1,000,000    | **Saved: 560,000** |
| **Potential Loss (if no model used)** | 1,880,000 | - | **Saved: 880,000** |
| **F2-Score Focus**       | Lower Recall, Higher Precision  | Higher Recall, Lower Precision | **Better Recall (Catching More Attrition Cases)** |

## 📉 Model Performance Comparison  

The model was optimized using **hyperparameter tuning** to prioritize **recall** (important for reducing attrition losses). The tuned model has improved recall, ensuring more at-risk employees are correctly identified.

| **Metric**     | **Before Hyperparameter Tuning** | **After Hyperparameter Tuning** |
|---------------|--------------------------------|-------------------------------|
| **Precision (Class 1 - Attrition)** | 1.00 | 0.38 |
| **Recall (Class 1 - Attrition)** | 0.17 | **0.79** |
| **F1-Score (Class 1 - Attrition)** | 0.29 | 0.51 |
| **Accuracy** | 0.87 | 0.76 |
| **Macro Avg Recall** | 0.59 | **0.77** |
| **Macro Avg F1-Score** | 0.61 | **0.68** |


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
