import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ==============================
# 1. Load Trained Model (.pkl)
# ==============================
MODEL_PATH = "best_tuned_MLP.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ==============================
# 2. Streamlit App UI
# ==============================
st.title("📊 Cardiovascular Disease Prediction App")
st.write("This interface predicts cardiovascular disease risk using statistical input data.")

st.sidebar.header("🔧 Input Features")

# Example numeric inputs — replace with your dataset's features
age = st.sidebar.number_input("Age (years)", min_value=20, max_value=100, value=45)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=220, value=165)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
ap_hi = st.sidebar.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
ap_lo = st.sidebar.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
cholesterol = st.sidebar.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
gluc = st.sidebar.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
smoke = st.sidebar.selectbox("Smoke", [0, 1], format_func=lambda x: "Yes" if x else "No")
alco = st.sidebar.selectbox("Alcohol", [0, 1], format_func=lambda x: "Yes" if x else "No")
active = st.sidebar.selectbox("Physical Activity", [0, 1], format_func=lambda x: "Yes" if x else "No")

# Combine features into DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'height': [height],
    'weight': [weight],
    'ap_hi': [ap_hi],
    'ap_lo': [ap_lo],
    'cholesterol': [cholesterol],
    'gluc': [gluc],
    'smoke': [smoke],
    'alco': [alco],
    'active': [active]
})

# ==============================
# 3. Prediction Section
# ==============================
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)
    probability = None
    try:
        probability = model.predict_proba(input_data)[0][1]
    except:
        pass

    st.subheader("🩺 Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Cardiovascular Disease")
    else:
        st.success("✅ Low Risk of Cardiovascular Disease")

    if probability is not None:
        st.write(f"Prediction Confidence: **{probability*100:.2f}%**")

    # ==============================
    # 4. Statistics Visualization
    # ==============================
    st.subheader("📈 Statistical Summary")
    st.dataframe(input_data.describe())

    st.subheader("📊 BMI Visualization")
    bmi = weight / ((height/100)**2)
    fig, ax = plt.subplots()
    ax.bar(["BMI"], [bmi])
    ax.axhline(25, color='r', linestyle='--', label='Overweight Threshold')
    ax.legend()
    st.pyplot(fig)

# ==============================
# 5. Upload Batch Data (Optional)
# ==============================
st.sidebar.write("---")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV for Batch Prediction", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    preds = model.predict(df)
    df['Prediction'] = preds
    st.write("### Batch Prediction Results")
    st.dataframe(df.head())

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
