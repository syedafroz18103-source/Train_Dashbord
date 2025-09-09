from utils.preprocessing import load_data, preprocessing
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load and preprocess dataset
df = load_data()
df1 = preprocessing(df)


st.title("🔗 Correlations,drivers & interactive slice-and-dice ")

# --- Feature engineering ---
df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED'].clip(upper=0) / -365.25

numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
numeric_df['AGE_YEARS'] = df['AGE_YEARS']
numeric_df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS']

# --- Correlations ---
corr_series = numeric_df.corr()['TARGET'].drop('TARGET').sort_values(ascending=False)

top5_pos = corr_series[corr_series > 0].nlargest(5)
top5_neg = corr_series.nsmallest(5)

corr_with_income = numeric_df.corr()['AMT_INCOME_TOTAL'].drop('AMT_INCOME_TOTAL').abs().sort_values(ascending=False)
most_corr_income = corr_with_income.max()

corr_with_credit = numeric_df.corr()['AMT_CREDIT'].drop('AMT_CREDIT').abs().sort_values(ascending=False)
most_corr_credit = corr_with_credit.max()

corr_income_credit = numeric_df['AMT_INCOME_TOTAL'].corr(numeric_df['AMT_CREDIT'])
corr_age_target = numeric_df['AGE_YEARS'].corr(numeric_df['TARGET'])
corr_emp_target = numeric_df['EMPLOYMENT_YEARS'].corr(numeric_df['TARGET'])
family_col = 'CNT_FAM_MEMBERS' if 'CNT_FAM_MEMBERS' in numeric_df.columns else None
corr_family_target = numeric_df[family_col].corr(numeric_df['TARGET']) if family_col else np.nan

abs_corr = corr_series.abs().sort_values(ascending=False)
top5_features = abs_corr.index[:5]
variance_explained_proxy = (corr_series[top5_features] ** 2).sum()

high_corr_count = (corr_series.abs() > 0.5).sum()

# --- Streamlit UI ---
st.title("Correlation Insights & KPIs")

# KPIs (metrics)
col1, col2, col3, col4= st.columns(4)
col1.metric("Most correlated with Income", most_corr_income)
col2.metric("Most correlated with Credit", most_corr_credit)
col3.metric("Corr(Income, Credit)", round(corr_income_credit, 4))
col4.metric("Corr(Age, TARGET)", round(corr_age_target, 4))
col5, col6, col7,col8 = st.columns(4)
col5.metric("Corr(Employment Years, TARGET)", round(corr_emp_target, 4))
col6.metric("Corr(Family Size, TARGET)", round(corr_family_target, 4))
col7.metric("Variance explained (Top 5 R² proxy)", round(variance_explained_proxy, 4))
col8.metric("# Features with |corr| > 0.5", int(high_corr_count))


col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 5 Positive Correlations with TARGET")
    st.table(top5_pos)

with col2:
    st.subheader("Top 5 Negative Correlations with TARGET")
    st.table(top5_neg)

c1,c2 = st.columns(2)
with c1:
#Heatmap — Correlation (selected numerics)
    st.subheader("Correlation (selected numerics)")
    plt.figure(figsize=(3,3))
    sns.heatmap(numeric_df[['TARGET','AGE_YEARS','EMPLOYMENT_YEARS','AMT_INCOME_TOTAL','AMT_CREDIT']].corr(),annot=True, fmt=".2f", cmap="viridis", vmin=-1, vmax=1)
    st.pyplot(plt)

with c2:
#Bar — |Correlation| of features vs TARGET (top N)
    st.subheader("|Correlation| of features vs TARGET (top N)")
    plt.figure(figsize=(3,3))
    plt.bar(corr_series.index, corr_series.values, color="#7d9be5")
    plt.title("Distribution of Correlations with TARGET")
    plt.xlabel("Correlation with TARGET")
    plt.ylabel("Feature Count")
    st.pyplot(plt)


c1,c2=st.columns(2)
with c1:
#Scatter — Age vs Credit (hue=TARGET)
    st.subheader("Age vs Credit (hue=TARGET)")
    plt.figure(figsize=(3,3))
    sns.scatterplot(data=df1, x="AGE_YEARS", y="AMT_CREDIT", hue="TARGET", alpha=0.6, palette=["#24D0B3", "#FF6B6B"])
    plt.title("Age vs Credit by Target")
    plt.xlabel("Age (Years)")
    plt.ylabel("Credit Amount")
    plt.legend(title="TARGET")
    st.pyplot(plt)

with c2:
#Scatter — Age vs Income (hue=TARGET)
    st.subheader("Age vs Income (hue=TARGET)")
    plt.figure(figsize=(3,3))
    sns.scatterplot(data=df1, x="AGE_YEARS", y="AMT_INCOME_TOTAL", hue="TARGET", alpha=0.6, palette=["#604BE8", "#5E2E2E"])
    plt.title("Age vs Credit by Target")
    plt.xlabel("Age (Years)")
    plt.ylabel("Income Amount")
    plt.legend(title="TARGET")
    st.pyplot(plt)

c1,c2=st.columns(2)
with c1:
#Scatter — Employment Years vs TARGET (jitter/bins)
    st.subheader("Employment Years vs TARGET (with jitter)")
    plt.figure(figsize=(3,3))
    df1_jitter = df1.copy()
    df1_jitter["TARGET_JITTER"] = df1_jitter["TARGET"] + np.random.uniform(-0.1, 0.1, size=len(df1_jitter))
    sns.scatterplot(data=df1_jitter,x="EMPLOYMENT_YEARS",y="TARGET_JITTER",hue="TARGET",alpha=0.6,palette=["#D58BD8", "#ECE581"],)
    plt.title("Employment Years vs TARGET (with jitter)")
    plt.xlabel("Employment Years")
    plt.ylabel("TARGET")
    plt.yticks([0,1],["Repaid","Default"])
    plt.legend(title="TARGET")
    st.pyplot(plt)

with c2:
#Boxplot — Credit by Education
    st.subheader("Credit by Education")
    plt.figure(figsize=(3,3))
    plt.title("Credit by Education")
    sns.boxplot(data=df1,x="NAME_EDUCATION_TYPE",y="AMT_CREDIT",palette="Set2")
    plt.title("Credit Distribution by Education")
    plt.xlabel("Education Level")
    plt.xticks(rotation=90)
    plt.ylabel("Credit Amount")
    st.pyplot(plt)

c1,c2=st.columns(2)
with c1:
#Boxplot — Income by Family Status
    st.subheader("Income by Family Status")
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df1,x="NAME_FAMILY_STATUS",y="AMT_INCOME_TOTAL",palette="Set3")
    plt.title("Income by Family Status")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(plt)

with c2:
#Pair Plot — Income, Credit, Annuity, TARGET
    st.subheader("Income, Credit, Annuity, TARGET")
    plt.figure(figsize=(3,3))
    plt.title("Pair Plot — Income, Credit, Annuity, TARGET")
    sns.pairplot(df1[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "TARGET"]],hue="TARGET",diag_kind="kde",palette={0: "#3F2454", 1: "#6BA65F"})
    st.pyplot(plt)

c1,c2=st.columns(2)
with c1:
#Filtered Bar — Default Rate by Gender (responsive to sidebar)
    st.header("Default Rate by Gender (responsive to sidebar)")
    plt.figure(figsize=(3,3))
    default_rate = df1.groupby("CODE_GENDER")["TARGET"].mean().reset_index()
    default_rate["TARGET"] = default_rate["TARGET"] * 100
    sns.barplot(data=default_rate,x="CODE_GENDER",y="TARGET",palette=["#465D97", "#863E3E"])
    plt.title("Filtered Bar — Default Rate by Gender (responsive to sidebar)")
    plt.ylabel("Default Rate (%)")
    plt.xlabel("Gender")
    st.pyplot(plt)

with c2:
#Filtered Bar — Default Rate by Education (responsive)
    plt.figure(figsize=(6,4))
    default_rate = df1.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().reset_index()
    default_rate["TARGET"] = default_rate["TARGET"] * 100
    sns.barplot(data=default_rate, x="NAME_EDUCATION_TYPE",y="TARGET",palette="Set2")
    plt.ylabel("Default Rate (%)")
    plt.xlabel("Education Level")
    plt.title("Default Rate by Education")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(plt)

