import pandas as pd
import numpy as np
import streamlit as st


def load_data(filepath="./application_train.csv"):
    df=pd.read_csv(filepath)
    return df

def preprocessing(df):
    df["AGE_YEARS"] = -(df["DAYS_BIRTH"] / 365.25)
    df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EMPLOYMENT_YEARS"] = df["EMPLOYMENT_YEARS"].clip(upper=100)
    df["DTI"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["LOAN_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_TO_CREDIT"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

  #4. Handle missing values (report % and apply strategy: drop columns > 60% missing; impute median/most-frequent for others)
    missing_ratio=df.isna().mean()
    drop_cols = missing_ratio[missing_ratio > 0.6].index
    df = df.drop(columns=drop_cols)

    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0])
            else:
                df[col].fillna(df[col].median())

# 5. Standardize categories (merge rare categories under “Other” if share < 1%)
    for col in df.select_dtypes(include="object"):
        freqs = df[col].value_counts(normalize=True)
        rare = freqs[freqs < 0.01].index
        df[col] = df[col].replace(rare, "Other")


#6.Outlier handling: Winsorize top/bottom 1% for skewed numeric features used in charts
    for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)    


#7.Define income brackets (quantiles): Low (Q1), Mid (Q2–Q3), High (Q4)
    df["INCOME_BRACKET"] = pd.qcut(df["AMT_INCOME_TOTAL"], q=4, labels=["Low", "Mid-Low", "Mid-High", "High"])    
    return df

