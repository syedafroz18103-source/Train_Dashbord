import streamlit as st
from utils.preprocessing import load_data, preprocessing

st.set_page_config(page_title="Application Train(Loan)", page_icon="🏦", layout="wide")

st.title("🏦 Application train dashboard")

st.markdown("""
Welcome to the *Application train Dashboard* built with *Streamlit*.  
* 📊 Overview & data quality 
* 🎯 Target & risk management 
* 👨‍👩‍👧‍👦 Demographics & Household profile  
* 📈 Financial health & Affordability  
* 🔗 Correlations,drivers & interactive slice-and-dice
""")


st.markdown("---")
st.subheader("Upload / Use Default Dataset")

uploaded_file = st.file_uploader("Upload your application train CSV file", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data()

processed_df=preprocessing(df)
st.dataframe(processed_df.head(10))

