import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval / Credit Risk Prediction")
st.write("Random Forest based ML Web App")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("credit_risk_dataset.csv")

    # Handle missing values
    df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
    df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    cat_cols = [
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ]
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    return df

df = load_data()

# -----------------------------
# Train Model
# -----------------------------
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Applicant Details")

person_age = st.sidebar.number_input("Age", 18, 100, 30)
person_income = st.sidebar.number_input("Annual Income", 0, 1000000, 50000)
person_home_ownership = st.sidebar.selectbox(
    "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
)
person_emp_length = st.sidebar.number_input("Employment Length (years)", 0.0, 50.0, 5.0)
loan_intent = st.sidebar.selectbox(
    "Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)
loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("Loan Amount", 0, 500000, 10000)
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", 0.0, 50.0, 10.0)
loan_percent_income = st.sidebar.slider("Loan % of Income", 0.0, 1.0, 0.2)
cb_person_default_on_file = st.sidebar.selectbox("Previous Default", ["N", "Y"])
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (years)", 0, 50, 5)

# -----------------------------
# Encode Inputs (Manual Mapping)
# -----------------------------
home_map = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
intent_map = {
    "PERSONAL": 0, "EDUCATION": 1, "MEDICAL": 2,
    "VENTURE": 3, "HOMEIMPROVEMENT": 4, "DEBTCONSOLIDATION": 5
}
grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
default_map = {"N": 0, "Y": 1}

input_data = pd.DataFrame([[
    person_age,
    person_income,
    home_map[person_home_ownership],
    person_emp_length,
    intent_map[loan_intent],
    grade_map[loan_grade],
    loan_amnt,
    loan_int_rate,
    loan_percent_income,
    default_map[cb_person_default_on_file],
    cb_person_cred_hist_length
]], columns=X.columns)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
