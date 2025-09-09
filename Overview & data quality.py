from utils.preprocessing import load_data, preprocessing
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load and preprocess dataset
df =load_data()
df1 = preprocessing(df)



st.title("📊 Overview & Data Quality Dashboard")

# ====================== KPIs ==========================
Total_Applicants = df1['SK_ID_CURR'].count()
Default_Rate = df1['TARGET'].mean() * 100
Repaid_Rate = (1 - df1['TARGET'].mean()) * 100
Total_Features = df1.shape[1]
Avg_Missing_per_Feature = df1.isnull().mean().mean() * 100
Numerical_Features = df1.select_dtypes(include=np.number).shape[1]
Categorical_Features = df1.select_dtypes(include='object').shape[1]
Median_Age = int(df1['AGE_YEARS'].median())
Median_Annual_Income = df1['AMT_INCOME_TOTAL'].median()
Average_Credit_Amount = df1['AMT_CREDIT'].mean()

# ====================== Display KPIs ==========================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Applicants", f"{Total_Applicants:,}")
col2.metric("Default Rate (%)", f"{Default_Rate:.2f}%")
col3.metric("Repaid Rate (%)", f"{Repaid_Rate:.2f}%")
col4.metric("Total Features", f"{Total_Features}")
col5.metric("Avg Missing per Feature (%)", f"{Avg_Missing_per_Feature:.2f}%")

col6, col7, col8, col9, col10 = st.columns(5)
col6.metric("Numerical Features", Numerical_Features)
col7.metric("Categorical Features", Categorical_Features)
col8.metric("Median Age (Years)", f"{Median_Age}")
col9.metric("Median Annual Income", f"{Median_Annual_Income}")
col10.metric("Average Credit Amount", f"{Average_Credit_Amount}")

st.markdown("---")


c1, c2 = st.columns(2)
# Pie / Donut — Target distribution (0 vs 1)
with c1:
    st.subheader("🎯 Target Distribution")
    target=df1['TARGET'].value_counts()
    st.pyplot(target.plot.pie(autopct="%1.1f%%", figsize=(5,4),ylabel="").get_figure())
with c2:
#Bar — Top 20 features by missing %
    st.subheader("Top 20 features by missing %")
    missing_p=df1.isna().mean()*100
    sort=missing_p.sort_values(ascending=False).head(20)
    st.bar_chart(sort)

c1, c2 = st.columns(2)
with c1:
# Histogram — AGE_YEARS
    st.subheader("Histogram AGE_YEARS")
    plt.figure(figsize=(6,6))
    plt.hist(df1['AGE_YEARS'],bins=10,color='#b3ff99',edgecolor='black')
    plt.xlabel("AGE_YEARS")
    plt.ylabel('COUNT')
    st.pyplot(plt)
with c2:
# Histogram — AMT_INCOME_TOTAL
    st.subheader("Histogram AMT_INCOME_TOTAL")
    plt.figure(figsize=(6,6))
    plt.hist(df1['AMT_INCOME_TOTAL'],bins=30,color='#bf80ff',edgecolor='black')
    plt.xlabel("AMT_INCOME_TOTAL")
    plt.ylabel('COUNT')
    st.pyplot(plt)

c1, c2 = st.columns(2)
with c1:
# Histogram — AMT_CREDIT
    st.subheader("AMT_CREDIT")
    plt.figure(figsize=(6,6))
    plt.hist(df1['AMT_CREDIT'],bins=20,color='g',edgecolor='black')
    plt.xlabel("AMT_CREDIT")
    plt.ylabel('COUNT')
    st.pyplot(plt)
with c2:
# Boxplot — AMT_INCOME_TOTAL
    st.subheader("AMT_INCOME_TOTAL")
    plt.figure(figsize=(6,6))
    plt.boxplot(df1['AMT_INCOME_TOTAL'], patch_artist=True, boxprops=dict(facecolor='#ff8000', color='black'),
                whiskerprops=dict(color='b'), capprops=dict(color='b'), medianprops=dict(color='red'))
    plt.ylabel("AMT_INCOME_TOTAL")
    st.pyplot(plt)
c1, c2 = st.columns(2)
with c1:
#Boxplot — AMT_CREDIT
    st.subheader("AMT_CREDIT")
    plt.figure(figsize=(6,6))
    plt.boxplot(df1['AMT_CREDIT'], patch_artist=True, boxprops=dict(facecolor='g'),
                whiskerprops=dict(color='#1a75ff'), capprops=dict(color='#1a75ff'), medianprops=dict(color='orange'))
    plt.ylabel("AMT_CREDIT")
    st.pyplot(plt)
with c2:
# Countplot — CODE_GENDER
    st.subheader("CODE_GENDER")
    plt.figure(figsize=(6,6))
    sns.countplot(x=df1["CODE_GENDER"], palette='mako') 
    plt.xlabel("CODE_GENDER")
    plt.ylabel("Count")     
    st.pyplot(plt)

c1, c2 = st.columns(2)
with c1:
# Countplot — NAME_FAMILY_STATUS
    st.subheader("NAME_FAMILY_STATUS")
    plt.figure(figsize=(6,6))
    sns.countplot(x=df1["NAME_FAMILY_STATUS"],palette='crest') 
    plt.xlabel("NAME_FAMILY_STATUS")
    plt.xticks(rotation=90)
    plt.ylabel("Count")     
    st.pyplot(plt)
with c2:
#Countplot — NAME_EDUCATION_TYPE
    st.subheader("NAME_EDUCATION_TYPE")
    plt.figure(figsize=(6,6))
    sns.countplot(x=df1["NAME_EDUCATION_TYPE"],palette='flare') 
    plt.xlabel("NAME_EDUCATION_TYPE")
    plt.xticks(rotation=90)
    plt.ylabel("Count")     
    st.pyplot(plt)


st.subheader("Quick Insights")
insights = []
if "TARGET" in df.columns:
    insights.append(f"- Filtered default rate: **{100*df['TARGET'].mean():.2f}%**.")

for s in insights:
    st.markdown(s)

st.caption("Tip: modify the global filters in the sidebar to slice the dataset. All visuals update accordingly.")