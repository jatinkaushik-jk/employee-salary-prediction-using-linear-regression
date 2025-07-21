import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("best_model.pkl")

# Load the original data to fit the LabelEncoders
# This assumes your original data with categorical columns is available
# If not, you would need to save and load the fitted encoders separately
try:
    original_data = pd.read_csv(r"/content/final_data.csv")
except FileNotFoundError:
    st.error("Error: original_data.csv not found. Please make sure the file exists in the correct path.")
    st.stop()

# Fit LabelEncoders for each categorical column used in training
education_encoder = LabelEncoder()
occupation_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

# Filter out '?' and replace with 'Others' for occupation
education_classes = [c for c in original_data['education'].unique() if c != '?']
occupation_classes = [c if c != '?' else 'Others' for c in original_data['occupation'].unique()]
gender_classes = [c for c in original_data['gender'].unique() if c != '?']


education_encoder.fit(education_classes)
occupation_encoder.fit(occupation_classes)
gender_encoder.fit(gender_classes)


st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("Input Employee Details")

# ✨ Replace these fields with your dataset's actual input columns
age = st.sidebar.slider("Age", 17, 75, 30) # Adjusted age range based on data cleaning
# Use the fitted encoders to get the list of classes for the selectbox
education = st.sidebar.selectbox("Education Level", education_classes)
gender = st.sidebar.selectbox("Gender", gender_classes)
occupation = st.sidebar.selectbox("Job Role", occupation_classes)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40) # Adjusted hours per week range based on data

# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'gender' : [gender],
})

# Preprocess the input data using the fitted encoders
input_df['education'] = education_encoder.transform(input_df['education'])
input_df['occupation'] = occupation_encoder.transform(input_df['occupation'])
input_df['gender'] = gender_encoder.transform(input_df['gender'])


st.write("### 🔎 Input Data (after preprocessing)")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    # Apply the same preprocessing to batch data
    try:
        batch_data['education'] = education_encoder.transform(batch_data['education'])
        # Handle potential '?' values in occupation column if they might be in the uploaded file
        batch_data['occupation'] = batch_data['occupation'].replace('?', 'Others')
        batch_data['occupation'] = occupation_encoder.transform(batch_data['occupation'])
        batch_data['gender'] = gender_encoder.transform(batch_data['gender'])
    except ValueError as e:
        st.error(f"Error during batch data preprocessing: {e}. Please ensure the uploaded CSV contains valid data for education, occupation, and gender based on the training data.")
        st.stop()


    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
