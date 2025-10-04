# End-to-End ML Loan Predictor

A machine learning web application that predicts loan approval probability using multiple algorithms including Random Forest, XGBoost, SVM, and Ensemble methods, deployed with Streamlit.

## 🚀 Features

- **Multiple ML Models**: Random Forest, XGBoost, SVM, Logistic Regression, and Ensemble
- **Web Interface**: User-friendly Streamlit web application
- **Feature Engineering**: Comprehensive data preprocessing pipeline
- **Model Interpretation**: Detailed analysis of predictions and key features
- **Performance Comparison**: Side-by-side model evaluation
- **Probability Prediction**: Returns loan approval probability scores

## 📊 Model Performance

### 🏆 Best Performing Model: Random Forest
| Metric | Score |
|--------|-------|
| **Accuracy** | 82.11% |
| **ROC-AUC** | 0.8432 |
| **Precision** | 0.8621 |
| **Recall** | 0.8824 |
| **F1-Score** | 0.8721 |

### 📈 Complete Model Comparison
| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|-------|----------|---------|-----------|--------|----------|
| **Random Forest** | 82.11% | 0.8432 | 0.8621 | 0.8824 | 0.8721 |
| **Ensemble** | 82.93% | 0.8393 | 0.8636 | 0.8941 | 0.8786 |
| **XGBoost** | 81.30% | 0.8331 | 0.8690 | 0.8588 | 0.8639 |
| **Logistic Regression** | 81.30% | 0.8003 | 0.8370 | 0.9059 | 0.8701 |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup
1. Clone the repository:
git clone https://github.com/soumyadeep115/end-to-end-ml-loan-predictor
cd end-to-end-ml-loan-predictor

2. Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

## Running Web Application
streamlit run app/main.py

## 🔍 Key Features Influencing Predictions
The model identifies these as most important factors:<br>
Credit_History - Most significant predictor<br>
ApplicantIncome - Applicant's income level<br>
LoanAmount - Requested loan amount<br>
DebtToIncomeRatio - Debt-to-income ratio<br>

## 📊 Dataset Information<br>
The project uses loan application data with features including:<br>
Applicant income and employment details<br>
Credit history and score<br>
Credit history and score<br>
Loan amount and term<br>
Property area and type<br>
Co-applicant information<br>
Debt-to-income ratios<br>

## 🤝 Contributing<br>
Fork the repository<br>
Create a feature branch (git checkout -b feature/AmazingFeature)<br>
Commit your changes (git commit -m 'Add some AmazingFeature')<br>
Push to the branch (git push origin feature/AmazingFeature)<br>
Open a Pull Request<br>