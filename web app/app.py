import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Set page config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# Create a demo model that trains instantly
def create_demo_model():
    # Create a simple but realistic demo model
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    
    # Create realistic training data based on common loan patterns
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic features
    applicant_income = np.random.normal(5000, 2000, n_samples)
    coapplicant_income = np.random.normal(2000, 1500, n_samples)
    loan_amount = np.random.normal(150000, 50000, n_samples)
    credit_history = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    loan_term = np.random.choice([360, 180, 120, 240], n_samples)
    
    # Create derived features
    total_income = applicant_income + coapplicant_income
    debt_to_income = loan_amount / (total_income + 1)
    emi = (loan_amount * 0.09/12 * (1 + 0.09/12)**loan_term) / ((1 + 0.09/12)**loan_term - 1)
    emi_to_income = emi / (applicant_income + 1)
    
    # Combine features
    X_demo = np.column_stack([
        applicant_income, coapplicant_income, loan_amount, credit_history,
        loan_term, total_income, debt_to_income, emi, emi_to_income
    ])
    
    # Create realistic target variable (loan approval)
    # Approval is more likely with higher income, good credit, lower DTI
    approval_score = (
        (applicant_income > 4000) * 2 +
        (coapplicant_income > 1000) * 1 +
        (credit_history == 1) * 3 +
        (debt_to_income < 10) * 2 +
        (emi_to_income < 0.5) * 2
    )
    y_demo = (approval_score >= 5).astype(int)
    
    # Train the model
    model.fit(X_demo, y_demo)
    return model, StandardScaler().fit(X_demo)

# Initialize demo model and scaler
demo_model, demo_scaler = create_demo_model()
feature_names = [
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History',
    'Loan_Amount_Term', 'TotalIncome', 'DebtToIncomeRatio', 'EMI', 'EMI_to_Income_Ratio'
]

def predict_loan_approval(user_inputs):
    """Make prediction using demo model"""
    try:
        # Prepare input features
        input_features = np.array([[
            user_inputs['ApplicantIncome'],
            user_inputs['CoapplicantIncome'], 
            user_inputs['LoanAmount'],
            user_inputs['Credit_History'],
            user_inputs['Loan_Amount_Term'],
            user_inputs['TotalIncome'],
            user_inputs['DebtToIncomeRatio'],
            user_inputs['EMI'],
            user_inputs['EMI_to_Income_Ratio']
        ]])
        
        # Scale features and predict
        input_scaled = demo_scaler.transform(input_features)
        probability = demo_model.predict_proba(input_scaled)[0, 1]
        prediction = demo_model.predict(input_scaled)[0]
        
        return prediction, probability
        
    except Exception as e:
        # Fallback to rule-based prediction if model fails
        st.warning(f"Using fallback prediction: {e}")
        return fallback_prediction(user_inputs)

def fallback_prediction(user_inputs):
    """Rule-based fallback prediction"""
    score = 0
    
    # Credit history is most important
    if user_inputs['Credit_History'] == 1:
        score += 30
    else:
        score += 5
        
    # Income factors
    if user_inputs['ApplicantIncome'] > 6000:
        score += 25
    elif user_inputs['ApplicantIncome'] > 4000:
        score += 15
    else:
        score += 5
        
    # Debt-to-income ratio
    if user_inputs['DebtToIncomeRatio'] < 5:
        score += 20
    elif user_inputs['DebtToIncomeRatio'] < 10:
        score += 10
    else:
        score += 5
        
    # EMI affordability
    if user_inputs['EMI_to_Income_Ratio'] < 0.4:
        score += 15
    elif user_inputs['EMI_to_Income_Ratio'] < 0.6:
        score += 8
    else:
        score += 2
        
    probability = min(score / 95, 0.95)
    prediction = 1 if probability > 0.6 else 0
    
    return prediction, probability

def main():
    st.title("🏦 Advanced Loan Approval Predictor")
    st.markdown("""
    This AI-powered application predicts loan approval probability based on applicant information.
    The model uses machine learning algorithms to provide accurate predictions.
    """)
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Unmarried", "Married"])
        dependents = st.slider("Number of Dependents", 0, 5, 0)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
        
        st.subheader("Credit Information")
        credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    with col2:
        st.subheader("Financial Information")
        
        applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input("Co-applicant Monthly Income ($)", min_value=0, value=2000, step=100)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=150000, step=1000)
        loan_term = st.slider("Loan Term (months)", 12, 480, 360)
        
        # Calculate derived features
        total_income = applicant_income + coapplicant_income
        debt_to_income_ratio = loan_amount / (total_income + 1e-8)
        
        # EMI Calculation
        monthly_interest = 0.09 / 12  # 9% annual interest
        emi = (loan_amount * monthly_interest * (1 + monthly_interest)**loan_term) / ((1 + monthly_interest)**loan_term - 1 + 1e-8)
        emi_to_income_ratio = emi / (applicant_income + 1e-8)
        
        # Display calculated values
        st.info(f"**Calculated Values:**")
        st.write(f"Total Income: ${total_income:,.0f}")
        st.write(f"Debt-to-Income Ratio: {debt_to_income_ratio:.1f}x")
        st.write(f"Estimated EMI: ${emi:,.2f}")

    # When predict button is clicked
    if st.button("🚀 Predict Loan Approval", use_container_width=True):
        # Prepare user inputs
        user_inputs = {
            'Gender': 1 if gender == "Male" else 0,
            'Married': 1 if married == "Married" else 0,
            'Dependents': dependents,
            'Education': 0 if education == "Graduate" else 1,
            'Self_Employed': 1 if self_employed == "Self-Employed" else 0,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': 1 if credit_history == "Good (1)" else 0,
            'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
            'Property_Area_Urban': 1 if property_area == "Urban" else 0,
            'TotalIncome': total_income,
            'DebtToIncomeRatio': debt_to_income_ratio,
            'EMI': emi,
            'EMI_to_Income_Ratio': emi_to_income_ratio
        }
        
        # Make prediction
        prediction, probability = predict_loan_approval(user_inputs)
        
        # Display results
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Create result columns
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            if prediction == 1:
                st.success(f"## ✅ LOAN APPROVED")
                st.balloons()
            else:
                st.error(f"## ❌ LOAN REJECTED")
        
        with res_col2:
            st.metric("Approval Probability", f"{probability:.2%}")
            
            # Confidence level
            if probability > 0.7:
                confidence = "High"
                color = "green"
            elif probability > 0.5:
                confidence = "Medium" 
                color = "orange"
            else:
                confidence = "Low"
                color = "red"
            
            st.metric("Confidence Level", confidence)
        
        with res_col3:
            st.metric("Monthly EMI", f"${emi:,.2f}")
            st.metric("EMI/Income Ratio", f"{emi_to_income_ratio:.1%}")
        
        # Progress bar
        st.progress(float(probability))
        
        # Detailed analysis
        st.subheader("📈 Detailed Analysis")
        
        anal_col1, anal_col2 = st.columns(2)
        
        with anal_col1:
            # Key metrics gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Approval Probability"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightcoral"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        with anal_col2:
            st.info("**Key Decision Factors:**")
            
            factors = [
                f"✅ Credit History: {'Excellent' if credit_history == 'Good (1)' else 'Needs Improvement'}",
                f"✅ Income Level: {'High' if applicant_income > 6000 else 'Medium' if applicant_income > 4000 else 'Low'}",
                f"✅ Debt-to-Income: {'Good' if debt_to_income_ratio < 5 else 'Acceptable' if debt_to_income_ratio < 10 else 'High'}",
                f"✅ EMI Affordability: {'Good' if emi_to_income_ratio < 0.4 else 'Manageable' if emi_to_income_ratio < 0.6 else 'High'}",
                f"✅ Employment: {'Stable' if self_employed == 'Salaried' else 'Variable'}"
            ]
            
            for factor in factors:
                st.write(factor)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit, Scikit-learn, and Random Forest</p>
        <p><small>Note: This is a demonstration application. Actual loan decisions involve additional factors.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()