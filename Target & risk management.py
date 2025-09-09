from utils.preprocessing import load_data, preprocessing
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load and preprocess dataset
df = load_data()
df1 = preprocessing(df)

st.title("🎯 Target & risk management Dashboard")


Total_default=df1['TARGET'].sum()
Default_Rate = df1['TARGET'].mean() * 100
Default_rate_by_gender=df1['CODE_GENDER'].value_counts(normalize=True)*100
D=Default_rate_by_gender.max()
Default_rate_by_Education=df1['NAME_EDUCATION_TYPE'].value_counts(normalize=True) * 100
d=Default_rate_by_Education.max()
Default_rate_by_Family_Status=df1['NAME_FAMILY_STATUS'].value_counts(normalize=True) * 100
d1=Default_rate_by_Family_Status.max()
Avg_Income=df1['AMT_INCOME_TOTAL'].mean()
Avg_Credit=df1['AMT_CREDIT'].mean()
Avg_Annuity=df1['AMT_ANNUITY'].mean()
Avg_Employment_years=(df1['DAYS_EMPLOYED']/365.25).mean()
Default_rate_by_housing_type=df1['NAME_HOUSING_TYPE'].value_counts(normalize=True)*100
E=Default_rate_by_housing_type.max()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Defaults",f"{Total_default}")
col2.metric("Default Rate (%)", f"{Default_Rate:.2f}%")
col3.metric("Default rate by gender(%)",f"{D:.2f}%")
col4.metric("Default rate by Education(%)",f"{d:.2f}%")
col5.metric("Default rate by family status(%)",f"{d1:.2f}%")

col6, col7, col8, col9, col10=st.columns(5)
col6.metric("Avg Income",f"{Avg_Income:.1f}")
col7.metric("Avg Credit",f"{Avg_Credit:.1f}")
col8.metric("Avg Annuity",f"{Avg_Annuity:.1f}")
col9.metric("Avg Employment(Years)",f"{Avg_Employment_years:.1f}")
col10.metric("Default Rate by Housing Type(%)",f"{E:.2f}%")



st.markdown("----")



st.subheader("Bar — Counts: Default vs Repaid")
plt.figure(figsize=(3,3))
sns.countplot(x='TARGET', data=df, order=[0,1])
plt.xlabel(['Repaid (0)', 'Default (1)'])
plt.title('Counts: Repaid vs Default')
st.pyplot(plt)


#Bar — Default % by Gender
st.subheader("Bar — Default % by Gender")
Bar_Default_by_gender=df1.groupby('CODE_GENDER')['TARGET'].mean()*100
plt.figure(figsize=(2,2))
plt.bar(Bar_Default_by_gender.index,Bar_Default_by_gender.values,color=["g","y","b"])
plt.title("Bar — Default % by Gender")
plt.xlabel("CODE_GENDER")
plt.ylabel("TARGET")
st.pyplot(plt)


#Bar — Default % by Education
st.subheader("Bar — Default % by Education")
default_rate = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean() * 100
default_rate = default_rate.sort_values(ascending=False)
plt.figure(figsize=(2,2))
plt.bar(default_rate.index,default_rate.values,color=["#4080bf"])
plt.title("Bar — Default % by Education")
plt.xticks(rotation=90)
plt.xlabel("NAME_EDUCATION_TYPE")
plt.ylabel("TARGET")
st.pyplot(plt)


#Bar — Default % by Family Status
st.subheader("Bar — Default % by Family Status")
default = df.groupby("NAME_FAMILY_STATUS")["TARGET"].mean() * 100
default = default.sort_values(ascending=False)
plt.figure(figsize=(2,2))
plt.bar(default.index,default.values,color=["#9B3434"])
plt.title("Bar — Default % by Family Status")
plt.xticks(rotation=90)
plt.xlabel("NAME_FAMILY_STATUS")
plt.ylabel("TARGET")
st.pyplot(plt)


#Bar — Default % by Housing Type
st.subheader("Bar — Default % by Housing Type")
default = df.groupby("NAME_HOUSING_TYPE")["TARGET"].mean() * 100
default = default.sort_values(ascending=False)
plt.figure(figsize=(2,2))
plt.bar(default.index,default.values,color=["#B0A1A1"])
plt.title("Bar — Default % by Housing Type")
plt.xticks(rotation=90)
plt.xlabel("NAME_HOUSING_TYPE")
plt.ylabel("TARGET")
st.pyplot(plt)


#Boxplot — Income by Target
st.subheader("Boxplot — Income by Target")
plt.figure(figsize=(3,3))
sns.boxplot(df1["AMT_INCOME_TOTAL"],patch_artist=True,boxprops=dict(facecolor='b'),
            whiskerprops=dict(color="#033B8E"), capprops=dict(color="#f778e4"), medianprops=dict(color='orange'))
plt.xlabel("TARGET (0 = Repaid, 1 = Default)")
plt.ylabel("INCOME")
st.pyplot(plt)


#Boxplot — Credit by Target
st.subheader("Boxplot — Credit by Target")
plt.figure(figsize=(3,3))
sns.boxplot(df1["AMT_CREDIT"],patch_artist=True,boxprops=dict(facecolor='g'),
            whiskerprops=dict(color="#C1DEAE"), capprops=dict(color="#3A7182"), medianprops=dict(color='y'))
plt.xlabel("TARGET (0 = Repaid, 1 = Default)")
plt.ylabel("CREDIT")
st.pyplot(plt)


#Violin — Age vs Target
st.subheader("Violin — Age vs Target")
plt.figure(figsize=(3,3))
sns.violinplot(x="TARGET", y="AGE_YEARS", data=df, inner="box", palette="mako")
plt.title("Age Distribution by Target")
plt.xlabel("Target (0 = Repaid, 1 = Default)")
plt.ylabel("Age (years)")
st.pyplot(plt)


#Histogram (stacked) — EMPLOYMENT_YEARS by Target
st.subheader("Histogram (stacked) — EMPLOYMENT_YEARS by Target")
plt.figure(figsize=(3,3))
repaid = df1.loc[df1["TARGET"] == 0, "EMPLOYMENT_YEARS"].dropna()
default = df1.loc[df1["TARGET"] == 1, "EMPLOYMENT_YEARS"].dropna()

plt.hist(
    [repaid, default],
    bins=30,
    stacked=True,
    label=["Repaid (0)", "Default (1)"],
    color=["skyblue", "salmon"],
    edgecolor="black"
)

plt.title("Employment Years by Target")
plt.xlabel("Employment Years")
plt.ylabel("Number of Clients")
st.pyplot(plt)


