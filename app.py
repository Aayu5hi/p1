import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.title("🚦 Traffic Speed Prediction System")

# Load dataset
df = pd.read_csv("traffic_dataset.csv")
st.write(df)
# Clean column names
df.columns = [
    "Timestamp",
    "IR Presence",
    "Vehicle Count",
    "Avg Speed (km/h)",
    "Vehicle Types Detected",
    "Vehicle Density (%)",
    "Saturation Flow Rate",
    "Volume Ratio",
    "FreeFlowSpeed (km/h)",
    "TSR",
    "VLSR",
    "Speed Factor",
    "CI",
    "Congestion Level"
]

df = df.dropna()

# Encode Congestion Level (since it's text)
le = LabelEncoder()
df["Congestion Level Encoded"] = le.fit_transform(df["Congestion Level"])

# Features (simple ones only)
features = [
    "Vehicle Count",
    "Vehicle Density (%)",
    "Congestion Level Encoded"
]

X = df[features]
y = df["Avg Speed (km/h)"]

# Train Regression Model
model = RandomForestRegressor()
model.fit(X, y)

st.success("Model Trained Successfully ✅")

# -------- USER INPUT --------

st.header("Enter Traffic Details")

vehicle_count = st.number_input("Vehicle Count", 0)
vehicle_density = st.number_input("Vehicle Density (%)", 0.0)

congestion_level = st.selectbox(
    "Congestion Level",
    le.classes_
)

if st.button("Predict Speed"):

    congestion_encoded = le.transform([congestion_level])[0]

    input_data = pd.DataFrame([[
        vehicle_count,
        vehicle_density,
        congestion_encoded
    ]], columns=features)

    prediction = model.predict(input_data)

    st.success(f"Predicted Average Speed: {round(prediction[0], 2)} km/h")
