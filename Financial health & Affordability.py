from utils.preprocessing import load_data, preprocessing
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load and preprocess dataset
df = load_data()
df1 = preprocessing(df)

st.title("📈 Financial health & Affordability ")


avg_income = df1["AMT_INCOME_TOTAL"].mean()
median_income = df1["AMT_INCOME_TOTAL"].median()
avg_credit = df1["AMT_CREDIT"].mean()
avg_annuity = df1["AMT_ANNUITY"].mean()
avg_goods = df1["AMT_GOODS_PRICE"].mean()
avg_dti = (df1["AMT_ANNUITY"] / df1["AMT_INCOME_TOTAL"]).mean()
avg_lti = (df1["AMT_CREDIT"] / df1["AMT_INCOME_TOTAL"]).mean()
inc_gap = (
    df1.loc[df1["TARGET"] == 0, "AMT_INCOME_TOTAL"].mean()
    - df1.loc[df1["TARGET"] == 1, "AMT_INCOME_TOTAL"].mean()
)
cred_gap = (
    df1.loc[df1["TARGET"] == 0, "AMT_CREDIT"].mean()
    - df1.loc[df1["TARGET"] == 1, "AMT_CREDIT"].mean()
)
pct_high_credit = (df1["AMT_CREDIT"] > 1_000_000).mean() * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Avg Annual Income", f"{avg_income:,.0f}")
col2.metric("Median Annual Income", f"{median_income:,.0f}")
col3.metric("Avg Credit Amount", f"{avg_credit:,.0f}")
col4.metric("Avg Annuity", f"{avg_annuity:,.0f}")
col5.metric("Avg Goods Price", f"{avg_goods:,.0f}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Avg DTI", f"{avg_dti:.2f}")
col2.metric("Avg LTI", f"{avg_lti:.2f}")
col3.metric("Income Gap (Non-def - Def)", f"{inc_gap:,.0f}")
col4.metric("Credit Gap (Non-def - Def)", f"{cred_gap:,.0f}")
col5.metric("% High Credit (>1M)", f"{pct_high_credit:.1f}%")


st.markdown("-----")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Histogram — Income distribution")
    plt.figure(figsize=(3,3))
    plt.title("Income distribution")
    plt.hist(df1["AMT_INCOME_TOTAL"].dropna(), bins=10,color="#5DC080",edgecolor="black")
    plt.xlabel("Annual Income")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(plt)

with c2:
    st.subheader("Histogram — Credit distribution")
    plt.figure(figsize=(3,3))
    plt.title("Credit distribution")
    plt.hist(df1["AMT_CREDIT"].dropna(), bins=10,color="#B8C05D",edgecolor="black")
    plt.xlabel("Credit Amount")
    plt.ylabel("Count")
    st.pyplot(plt)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Histogram — Annuity distribution")
    plt.figure(figsize=(3,3))
    plt.title("Annuity distribution")
    plt.hist(df1["AMT_ANNUITY"].dropna(), bins=10,color="#723743",edgecolor="black")
    plt.xlabel("Annuity Amount")
    plt.ylabel("Count")
    st.pyplot(plt)

with c2:
    st.subheader("Scatter — Income vs Credit (alpha blending)")
    plt.figure(figsize=(4,4))
    plt.title("Income vs Credit (alpha blending)")
    plt.scatter(df1["AMT_INCOME_TOTAL"], df1["AMT_CREDIT"],alpha=0.4, color="r")
    plt.xlabel("Income")
    plt.ylabel("Credit")
    st.pyplot(plt)

c1,c2=st.columns(2)
with c1:
    st.subheader("Scatter — Income vs Annuity")
    plt.figure(figsize=(4,4))
    plt.title("Income vs Annuity")
    plt.scatter(df1["AMT_INCOME_TOTAL"], df1["AMT_ANNUITY"],alpha=0.2, color="#FA05D5")
    plt.xlabel("Income")
    plt.ylabel("Annuity")
    st.pyplot(plt)
with c2:
    st.subheader("Boxplot — Credit by Target")
    plt.figure(figsize=(3,3))
    plt.title("Credit by Target")
    sns.boxplot(x=df1["TARGET"],y=df1["AMT_CREDIT"],boxprops=dict(facecolor='g'),
                whiskerprops=dict(color='#1a75ff'), capprops=dict(color='#1a75ff'), medianprops=dict(color='y'))
    plt.xlabel(["Repaid", "Default"])
    st.pyplot(plt)

c1,c=st.columns(2)
with c1:
#Boxplot — Income by Target
    st.subheader("Income by Target")
    plt.figure(figsize=(3,3))
    plt.title("Income by Target")
    sns.boxplot(x=df1["TARGET"],y=df1["AMT_INCOME_TOTAL"],boxprops=dict(facecolor='r'),
                whiskerprops=dict(color="#a3d039"), capprops=dict(color="#2D4365"), medianprops=dict(color='y'))
    plt.xlabel(["Repaid", "Default"])
    st.pyplot(plt)

with c2:
    st.subheader("Bar — Income Brackets vs Default Rate")
    plt.figure(figsize=(3,3))
    inc_br = df1.groupby("INCOME_BRACKET")["TARGET"].mean()
    plt.bar(inc_br.index, inc_br.values,color="#24D0B3")
    plt.xlabel("Income")
    plt.ylabel("Default Rate")
    st.pyplot(plt)

#Heatmap — Financial variable correlations (Income, Credit, Annuity, DTI, LTI, TARGET)
st.subheader("Heatmap — Financial variable correlations (Income, Credit, Annuity, DTI, LTI, TARGET)")
plt.figure(figsize=(3,3))
df1["DTI"]=df1["AMT_ANNUITY"] / df1["AMT_INCOME_TOTAL"]
df1["LTI"] = df1["AMT_CREDIT"] / df1["AMT_INCOME_TOTAL"]
cols = ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","DTI","LTI","TARGET"]
corr = df1[cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", cbar=False)
st.pyplot(plt)

st.subheader("Key Insights")
st.markdown("""
- Applicants with **higher Credit amounts relative to Income (high LTI)** 
  or **higher Annuity relative to Income (high DTI)** are at greater risk of default.
- High-credit applicants (>1M loan) form a small but risky segment.
- Overall, affordability metrics (DTI, LTI) are strong indicators of repayment capacity.
""")