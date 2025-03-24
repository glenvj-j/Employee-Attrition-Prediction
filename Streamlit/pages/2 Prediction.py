import pandas as pd
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="HR Attrition Prediction Tool",
    page_icon="📊",
    layout="wide"
)

# ==============
# Title
st.title("Employee Attrition Prediction & Risk Analysis")

# Load Cleaned Data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

df = pd.read_csv("https://raw.githubusercontent.com/glenvj-j/Employee-Attrition-Prediction/23d88e3b08e90d14a8c62917720f26fc12180973/Dataset/df_clean.csv")

def calculate_gaps(df):
    """
    Computes SalaryGap and DailyRateGap based on median values by Department and Job Level.
    """
    median_income = df.groupby(['Department', 'JobLevel'])['MonthlyIncome'].median().reset_index()
    median_income.rename(columns={'MonthlyIncome': 'MedianIncome'}, inplace=True)
    
    median_daily_rate = df.groupby(['Department', 'JobLevel'])['DailyRate'].median().reset_index()
    median_daily_rate.rename(columns={'DailyRate': 'MedianDailyRate'}, inplace=True)
    
    df = df.merge(median_income, how='left', on=['Department', 'JobLevel'])
    df = df.merge(median_daily_rate, how='left', on=['Department', 'JobLevel'])
    
    df['SalaryGap'] = df['MonthlyIncome'] - df['MedianIncome']
    df['DailyRateGap'] = df['DailyRate'] - df['MedianDailyRate']
    
    df.drop(columns=['MonthlyIncome', 'MedianIncome', 'DailyRate', 'MedianDailyRate'], inplace=True)
    return df

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_test' not in st.session_state:
    st.session_state.df_test = None

# Button to load data
if st.sidebar.button('Load Data'):
    st.session_state.df_test = load_data('https://raw.githubusercontent.com/glenvj-j/Employee-Attrition-Prediction/refs/heads/main/Dataset/sample_employee_data.csv')
    st.session_state.data_loaded = True

# Display data if loaded
if st.session_state.data_loaded and st.session_state.df_test is not None:
    st.sidebar.write("### Data Loaded Successfully!")
    df_test = st.session_state.df_test
    
    # Data preview before processing
    st.text(f'Preview of Employee Data | Total : {len(df_test)} Employees')

    st.dataframe(df_test.head().drop(columns=['Attrition']))
    df_test = df_test.drop(columns=['Attrition'])

    # Process Data
    df_test_a = calculate_gaps(df_test)
    X = df_test_a.drop(columns=['EmpID'])
    df_filtered = df_test_a.loc[:, df_test_a.columns.isin(X.columns)]

    # Load prediction model
    filename = 'https://raw.githubusercontent.com/glenvj-j/Employee-Attrition-Prediction/blob/main/HR%20Prediction%20Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    # Button to predict attrition
    if st.button('Run Prediction'):
        y_prob = loaded_model.predict_proba(df_filtered)[:, 1]
        threshold = 0.55
        df_test['Attrition'] = (y_prob >= threshold).astype(int)
        
        # Clustering for risk categorization
        X_2 = df_filtered[['JobSatisfaction','MonthlyRate']]
        scaler = StandardScaler()
        X_2_scaled = scaler.fit_transform(X_2)

        silhouette_scores = []
        number_of_cluster = range(2, 11)

        for i in number_of_cluster:
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X_2_scaled)
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(X_2_scaled, labels, metric='euclidean'))

        silhouette_df = pd.DataFrame({
            'Number of Clusters (K)': list(number_of_cluster),
            'Silhouette Score': silhouette_scores
        }).sort_values(by='Silhouette Score', ascending=False)

        kmeans = KMeans(n_clusters=int(silhouette_df.iloc[0]['Number of Clusters (K)']), random_state=42)
        kmeans.fit(X_2_scaled)
        df_test['cluster'] = kmeans.labels_

        # Mapping job satisfaction levels
        satisfaction_mapping = {1: 'Dissatisfied', 2: 'Neutral', 3: 'High Satisfaction', 4: 'Very Satisfied'}

        df_clustered = df_test.groupby('cluster').agg({
            'MonthlyRate': 'mean',
            'JobSatisfaction': 'median',
            'cluster': 'count'
        }).rename(columns={'cluster': 'Total'}).reset_index()
        df_clustered['PayCategory'] = pd.qcut(df_clustered['MonthlyRate'], q=3, labels=['Low', 'Mid', 'High'])
        df_clustered['Category'] = df_clustered['JobSatisfaction'].map(satisfaction_mapping)
        df_clustered['Label'] = df_clustered['PayCategory'].astype(str) + ' - ' + df_clustered['Category']

        # Assign Risk Levels
        def categorize_risk(row):
            if row['Category'] == 'Dissatisfied' or (row['PayCategory'] == 'Low' and row['Category'] == 'Neutral'):
                return 'High Risk'
            elif row['Category'] == 'Neutral' or (row['PayCategory'] == 'Mid' and row['Category'] == 'High Satisfaction'):
                return 'Medium Risk'
            else:
                return 'Low Risk'

        df_clustered['RiskLevel'] = df_clustered.apply(categorize_risk, axis=1)
        df_test['RiskLevel'] = df_test['cluster'].map(df_clustered.set_index('cluster')['RiskLevel'])

        high_risk = df_test[(df_test['RiskLevel'] == 'High Risk') & (df_test['Attrition'] == 1)]
        medium_risk = df_test[(df_test['RiskLevel'] == 'Medium Risk') & (df_test['Attrition'] == 1)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('⚠️ High-Risk Employees')
            st.text(f'Total Identified: {len(high_risk)} Employees')
            st.dataframe(high_risk[['EmpID', 'Department', 'JobRole', 'JobLevel']])
            high_csv = high_risk.to_csv(index=False).encode('utf-8')
            st.download_button("Download High-Risk Employees", high_csv, "high_risk.csv", "text/csv")

        with col2:
            st.subheader('🟡 Medium-Risk Employees')
            st.text(f'Total Identified: {len(medium_risk)} Employees')
            st.dataframe(medium_risk[['EmpID', 'Department', 'JobRole', 'JobLevel']])
            medium_csv = medium_risk.to_csv(index=False).encode('utf-8')
            st.download_button("Download Medium-Risk Employees", medium_csv, "medium_risk.csv", "text/csv")

        st.download_button("Download Complete Employee Data", df_test.to_csv(index=False).encode('utf-8'), "all_data.csv", "text/csv")

else:
    st.info("👈 Click 'Load Data'")
