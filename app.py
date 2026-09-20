import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Online Food Ordering Prediction", page_icon="🍔")

st.title("🍔 Online Food Ordering Prediction")
st.write("Predict whether a customer is likely to order food online.")

@st.cache_data
def load_data():
    return pd.read_csv("online food delivery dataset.csv")

df = load_data()

# Remove duplicate/unwanted column
if "Unnamed: 13" in df.columns:
    df = df.drop(columns=["Unnamed: 13"])

features = [
    "Age", "Gender", "Marital Status", "Occupation",
    "Monthly Income", "Educational Qualifications", "Family size",
    "Customer Type", "latitude", "longitude", "Pin code"
]

X = df[features].copy()
y = df["Output"].map({"Yes": 1, "No": 0})

# Handle missing values and categorical data
X = X.fillna(X.median(numeric_only=True))
X = pd.get_dummies(X, drop_first=True)
X = X.astype(float)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.sidebar.header("Customer Information")

age = st.sidebar.number_input("Age", 10, 100, 25)
gender = st.sidebar.selectbox("Gender", sorted(df["Gender"].dropna().unique()))
marital = st.sidebar.selectbox("Marital Status", sorted(df["Marital Status"].dropna().unique()))
occupation = st.sidebar.selectbox("Occupation", sorted(df["Occupation"].dropna().unique()))
income = st.sidebar.selectbox("Monthly Income", sorted(df["Monthly Income"].dropna().unique()))
education = st.sidebar.selectbox("Educational Qualifications", sorted(df["Educational Qualifications"].dropna().unique()))
family_size = st.sidebar.number_input("Family Size", 1, 20, 3)
customer_type = st.sidebar.selectbox("Customer Type", sorted(df["Customer Type"].dropna().unique()))
latitude = st.sidebar.number_input("Latitude", value=float(df["latitude"].median()))
longitude = st.sidebar.number_input("Longitude", value=float(df["longitude"].median()))
pin_code = st.sidebar.number_input("Pin Code", 100000, 999999, int(df["Pin code"].median()))

if st.button("🔮 Predict Order"):
    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Marital Status": [marital],
        "Occupation": [occupation],
        "Monthly Income": [income],
        "Educational Qualifications": [education],
        "Family size": [family_size],
        "Customer Type": [customer_type],
        "latitude": [latitude],
        "longitude": [longitude],
        "Pin code": [pin_code]
    })

    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("✅ Customer is predicted to ORDER food online.")
    else:
        st.warning("❌ Customer is predicted NOT to order food online.")

    st.info(f"Estimated probability of ordering: {probability:.2%}")

st.divider()
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("📌 Project Information")
st.write(
    "This project uses Logistic Regression, a linear classification algorithm, "
    "to predict whether a customer will order food online."
)
