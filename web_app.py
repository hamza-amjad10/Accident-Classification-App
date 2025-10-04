# web_app_full.py
import streamlit as st
import pandas as pd
import joblib

# Load saved pipeline
pipeline = joblib.load("final_pipeline.pkl")  

st.title("Accident Severity Prediction 🚦")

# --- Numeric Inputs ---
number_of_vehicles = st.number_input("Number of Vehicles Involved", min_value=1, max_value=10, value=2)
number_of_casualties = st.number_input("Number of Casualties", min_value=0, max_value=10, value=1)

# --- Categorical Inputs ---
Day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
Age_band_of_driver = st.selectbox("Age Band of Driver", ["18-30", "31-50", "Above 50"])
Sex_of_driver = st.selectbox("Sex of Driver", ["Male","Female"])
Educational_level = st.selectbox("Educational Level", ["Unknown","Below Secondary","Secondary","Higher"])
Vehicle_driver_relation = st.selectbox("Vehicle Driver Relation", ["Owner","Relative","Employee","Other"])
Driving_experience = st.selectbox("Driving Experience", ["<1yr","1-2yr","2-5yrs","5-10yrs","Above 10yr","Unknown"])
Type_of_vehicle = st.selectbox("Type of Vehicle", ["Automobile","Lorry (41?100Q)","Other","Pick up upto 10Q","Motorcycle"])
Owner_of_vehicle = st.selectbox("Owner of Vehicle", ["Private","Company","Government"])
Service_year_of_vehicle = st.selectbox("Service Year of Vehicle", ["Below 1yr","1-2yr","2-5yrs","5-10yrs","Above 10yr","Unknown"])
Defect_of_vehicle = st.selectbox("Defect of Vehicle", ["None","Brake","Steering","Lights","Other"])
Area_accident_occured = st.selectbox("Area Accident Occurred", ["Urban","Rural","Highway"])
Lanes_or_Medians = st.selectbox("Lanes or Medians", ["Two-way (divided with broken lines road marking)","Undivided Two way","Double carriageway (median)","Other"])
Road_allignment = st.selectbox("Road Alignment", ["Straight","Curve","Hill"])
Types_of_Junction = st.selectbox("Types of Junction", ["Cross","T-junction","Roundabout","Other"])
Road_surface_type = st.selectbox("Road Surface Type", ["Asphalt","Concrete","Gravel","Other"])
Road_surface_conditions = st.selectbox("Road Surface Conditions", ["Dry","Wet","Snow","Flood"])
Light_conditions = st.selectbox("Light Conditions", ["Daylight","Darkness - lights lit","Darkness - lights unlit"])
Weather_conditions = st.selectbox("Weather Conditions", ["Clear","Rain","Fog","Snow","Other"])
Type_of_collision = st.selectbox("Type of Collision", ["Rear-end","Head-on","Side","Other"])
Vehicle_movement = st.selectbox("Vehicle Movement", ["Going straight","Turning","Stopping","Other"])
Work_of_casuality = st.selectbox("Work of Casuality", ["Driver","Passenger","Pedestrian","Other"])
Fitness_of_casuality = st.selectbox("Fitness of Casuality", ["Fit","Unfit","Unknown"])
Casualty_class = st.selectbox("Casualty Class", ["Driver","Passenger","Pedestrian"])
Sex_of_casualty = st.selectbox("Sex of Casualty", ["Male","Female"])
Age_band_of_casualty = st.selectbox("Age Band of Casualty", ["0-18","19-30","31-50","Above 50"])
Casualty_severity = st.selectbox("Casualty Severity", ["Slight Injury","Serious Injury","Fatal injury"])
Pedestrian_movement = st.selectbox("Pedestrian Movement", ["Crossing","Walking along road","Other"])
Cause_of_accident = st.selectbox("Cause of Accident", ["Speeding","Overtaking","Distraction","Other"])
Time_hour = st.slider("Time (Hour of Accident)", 0, 23, 12)

# --- Create DataFrame from inputs ---
input_df = pd.DataFrame({
    "Number_of_vehicles_involved": [number_of_vehicles],
    "Number_of_casualties": [number_of_casualties],
    "Day_of_week": [Day_of_week],
    "Age_band_of_driver": [Age_band_of_driver],
    "Sex_of_driver": [Sex_of_driver],
    "Educational_level": [Educational_level],
    "Vehicle_driver_relation": [Vehicle_driver_relation],
    "Driving_experience": [Driving_experience],
    "Type_of_vehicle": [Type_of_vehicle],
    "Owner_of_vehicle": [Owner_of_vehicle],
    "Service_year_of_vehicle": [Service_year_of_vehicle],
    "Defect_of_vehicle": [Defect_of_vehicle],
    "Area_accident_occured": [Area_accident_occured],
    "Lanes_or_Medians": [Lanes_or_Medians],
    "Road_allignment": [Road_allignment],
    "Types_of_Junction": [Types_of_Junction],
    "Road_surface_type": [Road_surface_type],
    "Road_surface_conditions": [Road_surface_conditions],
    "Light_conditions": [Light_conditions],
    "Weather_conditions": [Weather_conditions],
    "Type_of_collision": [Type_of_collision],
    "Vehicle_movement": [Vehicle_movement],
    "Work_of_casuality": [Work_of_casuality],
    "Fitness_of_casuality": [Fitness_of_casuality],
    "Casualty_class": [Casualty_class],
    "Sex_of_casualty": [Sex_of_casualty],
    "Age_band_of_casualty": [Age_band_of_casualty],
    "Casualty_severity": [Casualty_severity],
    "Pedestrian_movement": [Pedestrian_movement],
    "Cause_of_accident": [Cause_of_accident],
    "Time_hour": [str(Time_hour)]
})

# --- Prediction ---
if st.button("Predict Accident Severity"):
    prediction = pipeline.predict(input_df)
    st.success(f"Predicted Accident Severity: {prediction[0]}")
