from utils.preprocessing import load_data, preprocessing
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load and preprocess dataset
df = load_data()
df1 = preprocessing(df)

st.title("👨‍👩‍👧‍👦 Demographics & Household profile ")


#"""% Male vs Female
gender_count = df1['CODE_GENDER'].value_counts(dropna=False)
percent_gender = (gender_count / gender_count.sum() * 100).round(2)
#Avg Age — Defaulters
avg_age_def = df1.loc[df1['TARGET'] == 1, 'AGE_YEARS'].mean()

avg_age_nondef = df1.loc[df1['TARGET'] == 0, 'AGE_YEARS'].mean()

percent_with_children = (df1['CNT_CHILDREN'].gt(0).mean() * 100).round(2)

avg_family_size = df1['CNT_FAM_MEMBERS'].mean()

family_count = df1['NAME_FAMILY_STATUS'].value_counts(dropna=False)
percent_family_status = (family_count / family_count.sum() * 100).round(2)

education_h = df1['NAME_EDUCATION_TYPE'].fillna('Unknown')
higher_mask = education_h.isin(['Higher education', 'Academic degree']) | education_h.str.contains('Bachelor|Master|Post', case=False, na=False)
percent_higher_education = (higher_mask.mean() * 100).round(2)

percent_with_parents = (df1['NAME_HOUSING_TYPE'].eq('With parents').mean() * 100).round(2)

is_working = df1['EMPLOYMENT_YEARS'].notna() & (df1['EMPLOYMENT_YEARS'] > 0)
is_working = is_working | df1['OCCUPATION_TYPE'].notna()
percent_currently_working = (is_working.mean() * 100).round(2)

avg_employment_years = df1.loc[df1['EMPLOYMENT_YEARS'].notna(), 'EMPLOYMENT_YEARS'].mean()

st.subheader("Gender Distribution (%)")
st.write(percent_gender.to_dict())

st.subheader("Family Status Distribution (%)")
st.write(percent_family_status.to_dict())
col2, col3,col4,col5 = st.columns(4)
col2.metric("Avg Age - Defaulters", round(avg_age_def, 2))
col3.metric("Avg Age - Non-Defaulters", round(avg_age_nondef, 2))
col4.metric("% With Children", percent_with_children)
col5.metric("Avg Family Size", round(avg_family_size, 2))

col7,col8,col9,col10= st.columns(4)
col7.metric("% Higher Education", percent_higher_education)
col8.metric("% Living With Parents", percent_with_parents)
col9.metric("% Currently Working", percent_currently_working)
col10.metric("Avg Employment Years", round(avg_employment_years, 2))



st.markdown("-----")

#Histogram — Age distribution (all)
st.subheader("Histogram — Age distribution (all)")
plt.figure(figsize=(3,3))
plt.hist(df1['AGE_YEARS'].dropna(),bins=10,color="#929598",edgecolor='black')
plt.xlabel("AGE_YEARS")
plt.ylabel('COUNT')
st.pyplot(plt)


#Histogram — Age by Target (overlay)
st.subheader("Histogram — Age by Target (overlay)")
plt.figure(figsize=(3,3))
plt.hist(df1.loc[df1['TARGET']==0,'AGE_YEARS'].dropna(), bins=10, label='Repaid (0)',color="#cc9966")
plt.hist(df1.loc[df1['TARGET']==1,'AGE_YEARS'].dropna(), bins=10, label='Default (1)',color="#1a8cff")
plt.xlabel("AGE")
plt.ylabel("TARGET")
plt.legend()
plt.title('Age Distribution by Target')
st.pyplot(plt)


#Bar — Gender distribution
st.subheader("Bar — Gender distribution")
plt.figure(figsize=(3,3))
CG=df1['CODE_GENDER'].value_counts()
plt.bar(CG.index,CG.values,color="#5c8a8a")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Gender distribution")
st.pyplot(plt)


#Bar — Family Status distribution
st.subheader("Bar — Family Status distribution")
plt.figure(figsize=(3,3))
FSD=df1['NAME_FAMILY_STATUS'].value_counts()
plt.bar(FSD.index,FSD.values,color="#ffff4d")
plt.title("Family Status distribution")
plt.xlabel("NAME_FAMILY_STATUS")
plt.xticks(rotation=90)
plt.ylabel("COUNT")
st.pyplot(plt)

#Bar — Education distribution
st.subheader("Bar — Education distribution")
plt.figure(figsize=(3,3))
ED=df1["NAME_EDUCATION_TYPE"].value_counts()
plt.bar(ED.index,ED.values,color="#34b74a")
plt.title("Education distribution")
plt.xlabel("NAME_EDUCATION_TYPE")
plt.xticks(rotation=90)
plt.ylabel("COUNT")
st.pyplot(plt)


#Bar — Occupation distribution (top 10)
st.subheader("Occupation distribution (top 10)")
plt.figure(figsize=(3,3))
OD=df1["OCCUPATION_TYPE"].value_counts().nlargest(10)
plt.bar(OD.index,OD.values,color="#8c3ab1")
plt.title("Occupation distribution (top 10)")
plt.xlabel("OCCUPATION_TYPE")
plt.ylabel("COUNT")
plt.xticks(rotation=90)
st.pyplot(plt)