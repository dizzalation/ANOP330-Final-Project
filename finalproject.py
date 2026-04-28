# Streamlit ML Web App: Bucknell Lending Club Loan Decision Tool
import streamlit as st
import pandas as pd
import joblib

# Load models and scalers
lasso_model = joblib.load('lasso_model.pkl')
log_model = joblib.load('log_model.pkl')
scaler_reg = joblib.load('scaler_reg.pkl')
scaler_clf = joblib.load('scaler_clf.pkl')
feature_names = joblib.load('feature_names.pkl')

# --- Streamlit UI ---
st.set_page_config(page_title="Bucknell Lending Club", layout="centered")
st.markdown("""
    <div style='background-color:#E87722; padding:25px; border-radius:10px; text-align:center; margin-bottom:20px'>
        <h1 style='color:white; margin:0; font-size:2.2em'> Bucknell Lending Club </h1>
        <p style='color:white; margin:5px 0 0 0; font-size:1.1em'>Loan Decision Support Tool | ANOP 330</p>
    </div>
""", unsafe_allow_html=True)
st.markdown("Enter a potential borrower's details to predict their risk of default and expected return.")

# User input
loan_amnt = st.number_input("Loan Amount ($)", 1000, 40000, 10000)
int_rate = st.slider("Interest Rate (%)", 5.3, 30.9, 13.0)
term_num = st.selectbox("Term (months)", [36, 60])
grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
annual_inc = st.number_input("Annual Income ($)", 10000, 500000, 65000)
dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0)
emp_length = st.selectbox("Employment Length", ["< 1 year","1 year","2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10+ years","Unknown"])
home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
verification_status = st.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"])
purpose = st.selectbox("Loan Purpose", ["debt_consolidation","credit_card","home_improvement","other","major_purchase","medical","small_business"])
fico = st.slider("FICO Score", 660, 850, 720)
revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 50.0)

# Prediction
if st.button("Predict"):
    raw = {
        'loan_amnt': loan_amnt,
        'int_rate': int_rate,
        'installment': loan_amnt * (int_rate / 1200) / (1 - (1 + int_rate / 1200) ** -term_num),
        'term_num': term_num,
        'annual_inc': annual_inc,
        'dti': dti,
        'fico_range_high': fico + 4,
        'fico_range_low': fico,
        'revol_util': revol_util,
        'revol_bal': 10000,
        'open_acc': 10,
        'pub_rec': 0,
        'delinq_2yrs': 0,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'verification_status': verification_status,
        'purpose': purpose,
        'grade': grade,
    }

    input_df = pd.DataFrame([raw])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    input_reg = scaler_reg.transform(input_df)
    input_clf = scaler_clf.transform(input_df)

    expected_return = lasso_model.predict(input_reg)[0]
    fully_paid_prob = log_model.predict_proba(input_clf)[0][1]
    default_prob = 1 - fully_paid_prob

    st.write(f"Predicted Pessimistic Return: **{expected_return:.2f}%**")
    st.write(f"Probability of Full Repayment: **{fully_paid_prob:.1%}**")

    if fully_paid_prob >= 0.75 and expected_return > 0:
        st.success("Recommendation: **FUND** — Low default risk and positive expected return.")
    elif fully_paid_prob >= 0.60 and expected_return > 0:
        st.warning("Recommendation: **CONSIDER** — Moderate risk, review carefully.")
    else:
        st.error("Recommendation: **DO NOT FUND** — High default risk or negative expected return.")

    probabilities = [fully_paid_prob, default_prob]
    labels = ['Fully Paid', 'Charged Off']
    chart_data = pd.DataFrame({'Probability': probabilities}, index=labels)
    st.write("Prediction Breakdown:")
    st.bar_chart(chart_data)

st.markdown("---")
st.markdown("""
    <div style='text-align:center; color:#E87722; font-weight:bold'>
        Bucknell Lending Club | ANOP 330 Final Project | Powered by Streamlit
    </div>
""", unsafe_allow_html=True)
