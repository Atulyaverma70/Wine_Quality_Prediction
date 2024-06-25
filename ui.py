import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load models
models = {
    "Logistic Regression": "lr_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "Artificial Neural Network": "ann_model.pkl",
    "Random Forest": "rf_model.pkl",
    "K-Nearest Neighbors": "knn_model.pkl",
    "Support Vector Machine": "svr_model.pkl",
    "Gradient Boosting Regressor": "gbr_model.pkl",
    "Naive Bayes": "nb_model.pkl",
    "Stochastic Gradient Descent": "sgd_model.pkl"
}

def load_model(model_file):
    return joblib.load(model_file)

# Load data
wine_data = pd.read_csv("WineQuality.csv")  # Replace with your dataset

# Data Preprocessing
labelencoder = LabelEncoder()
wine_data['Type'] = labelencoder.fit_transform(wine_data['Type'])

# Sidebar - Wine Details
st.sidebar.title("Wine Details")
fixed_acidity = st.sidebar.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.0)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.5)
citric_acid = st.sidebar.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.5)
residual_sugar = st.sidebar.number_input("Residual Sugar", min_value=0.0, max_value=16.0, value=6.0)
chlorides = st.sidebar.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.075)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=72.0, value=15.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=289.0, value=50.0)
density = st.sidebar.number_input("Density", min_value=0.0, max_value=2.0, value=0.997)
pH = st.sidebar.number_input("pH", min_value=2.0, max_value=5.0, value=3.0)
sulphates = st.sidebar.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.5)
alcohol = st.sidebar.number_input("Alcohol", min_value=0.0, max_value=100.0, value=9.0)
wine_type = st.sidebar.selectbox("Type", ["Red", "White"])

# Sidebar - Model Selection
st.sidebar.title("Model Selection")
selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))

# Function to convert wine type to numerical value
def get_wine_type(type_str):
    return 0 if type_str == "Red" else 1

# Function to predict wine quality
def predict_quality():
    st.write("Predicting quality...")

    # Convert wine type to numerical value
    wine_type_num = get_wine_type(wine_type)

    # Create DataFrame with selected wine details
    selected_wine_details = {
        "Fixed Acidity": fixed_acidity,
        "Volatile Acidity": volatile_acidity,
        "Citric Acid": citric_acid,
        "Residual Sugar": residual_sugar,
        "Chlorides": chlorides,
        "Free Sulfur Dioxide": free_sulfur_dioxide,
        "Total Sulfur Dioxide": total_sulfur_dioxide,
        "Density": density,
        "pH": pH,
        "Sulphates": sulphates,
        "Alcohol": alcohol/10,
        "Type": wine_type_num
    }

    selected_wine_df = pd.DataFrame([selected_wine_details])

    # Load selected model
    model = load_model(models[selected_model])

    # Predict quality
    quality_prediction = model.predict(selected_wine_df)

    # Scale prediction to a score out of 10
    score_out_of_10 = quality_prediction[0]

    # Display prediction result
    st.subheader("Predicted Wine Quality (Out of 10)")
    #write result bigger font
    st.write(f"# {score_out_of_10}")


# Run button
if st.sidebar.button("Run"):
    predict_quality()
