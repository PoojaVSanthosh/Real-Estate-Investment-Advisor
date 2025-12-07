import streamlit as st
import mlflow.pyfunc
import pandas as pd
import joblib

# ---------------------------
# Load Production Model + Vectorizer
# ---------------------------

model = "xgboost_model.pkl"
VECTORIZER_PATH = "dict_vectorizer.pkl"

@st.cache_resource
def load_model():
    model = joblib.load("xgboost_model.pkl")
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, dv = load_model()

st.title("🏠 Real Estate Price Predictor & Investment Recommender")

st.write("Enter property details below to estimate price & investment quality.")

# ---------------------------
# User Input Section
# ---------------------------

State = st.text_input("State")
City = st.text_input("City")

Property_Type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
Furnished_Status = st.selectbox("Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"])
Facing = st.selectbox("Facing Direction", ["North", "South", "East", "West"])
Owner_Type = st.selectbox("Owner Type", ["Owner", "Broker", "Builder"])
Availability_Status = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])

BHK = st.number_input("BHK", 1, 10)
Size_in_SqFt = st.number_input("Size (Sq Ft)", 300, 10000)
Year_Built = st.number_input("Year Built", 1970, 2025)
Floor_No = st.number_input("Floor Number", 0, 50)
Total_Floors = st.number_input("Total Floors", 1, 50)

Nearby_Schools = st.number_input("Nearby Schools (count)", 0, 20)
Nearby_Hospitals = st.number_input("Nearby Hospitals (count)", 0, 20)

Public_Transport_Accessibility = st.selectbox(
    "Public Transport Accessibility",
    ["Low", "Medium", "High"]
)

transport_map = {"Low": 1, "Medium": 2, "High": 3}

Parking_Space = st.selectbox("Parking Space", ["Yes", "No"])
Security = st.selectbox("Security", ["Yes", "No"])

Amenities = st.multiselect(
    "Amenities",
    ["Gym", "Pool", "Garden", "Clubhouse", "Playground"]
)

Amenities_Count = len(Amenities)

# ---------------------------
# Prediction Button
# ---------------------------

if st.button("Predict Price"):

    input_dict = {
        "State": State.lower(),
        "City": City.lower(),
        "Property_Type": Property_Type,
        "BHK": BHK,
        "Size_in_SqFt": Size_in_SqFt,
        "Year_Built": Year_Built,
        "Furnished_Status": Furnished_Status,
        "Floor_No": Floor_No,
        "Total_Floors": Total_Floors,
        "Age_of_Property": 2025 - Year_Built,
        "Nearby_Schools": Nearby_Schools,
        "Nearby_Hospitals": Nearby_Hospitals,
        "Public_Transport_Accessibility": transport_map[Public_Transport_Accessibility],
        "Parking_Space": Parking_Space,
        "Security": Security,
        "Amenities": ", ".join(Amenities) if Amenities else "None",
        "Facing": Facing,
        "Owner_Type": Owner_Type,
        "Availability_Status": Availability_Status,
        "Amenities_Count": Amenities_Count
    }

    # Convert to vectorizer format
    vectorized = dv.transform([input_dict])

    # Predict using MLflow model
    price_prediction = model.predict(vectorized)[0]

    st.subheader("🏷️ Estimated Property Price")
    st.success(f"₹ {round(price_prediction, 2)} Lakhs")

    # ---------------------------
    # Investment Recommendation
    # ---------------------------

    if price_prediction < 8:
        recommendation = "🟢 **Excellent Investment (High ROI Potential)**"
    elif 8 <= price_prediction < 14:
        recommendation = "🟡 **Moderate Investment (Stable Growth)**"
    else:
        recommendation = "🔴 **Risky Investment (High Price Zone)**"

    st.subheader("📊 Investment Recommendation")
    st.info(recommendation)
