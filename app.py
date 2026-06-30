# app.py 
import streamlit as st
import pickle 
import numpy as np

# Load model and Scaler 
with open('knn_model.pkl','rb') as f:
    model = pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

# Title and description 
st.set_page_config(page_title="Loan Approval Predictor",page_icon="💰")
st.title("💰 Loan Appproval Prediction")
st.markdown("Enter your finacial details below to see if your loan will be approved.")

# input fields in two columns for better layout 
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age",18,70,30)
    annual_income = st.slider("Annnual Income (in $1,000s)",20,200,60)
    credit_score = st.slider("Credit Score" , 300, 850, 650)
    loan_amount = st.slider("Loan Amount (in $1,000s)",5,100,30)

with col2:
    employement_year = st.slider("Year of employement",0,48,5)
    debt_to_income = st.slider("Debt-to-income Ratio",0.0,1.0,0.3,step=0.01)
    dependents = st.slider("Number of Dependents",0,5,1)

# Pridiction button
if st.button("predict Loan Approval",type="primary"):
    # Prepare input 2d array
    input_data = np.array([[age,annual_income,credit_score,loan_amount,employement_year,debt_to_income,dependents]])
    
    # Scale the input 
    input_scaled = scaler.transform(input_data)

    # Pridict class and probabilities
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0] #[prob_reject, prob_approve]
    
    # Display result 
    st.divider()
    if prediction == 1:
        st.success(f"Loan Approved! (Confidence :{proba[1]*100:.1f}%)")
    else:
        st.error(f" Loan Rejected. (Confidence: {proba[1]*100:.1f}%)")
        
        