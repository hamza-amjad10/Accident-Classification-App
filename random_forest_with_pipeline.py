import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

df=pd.read_csv("10 pipe dataset.csv")



# first convert time columns to hours
df["Time_hour"]=pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour.astype(str)

df.drop("Time",axis=1,inplace=True)


# now first fo train test split

X=df.drop("Accident_severity",axis=1)
Y=df["Accident_severity"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


# print(X_train)
# print(Y_train)



# now fill values 

numeric_features=["Number_of_vehicles_involved","Number_of_casualties"]

numeric_pipeline=Pipeline([
    ("scaler",StandardScaler())
])

Imputer_encoder_categorical_features=["Educational_level","Vehicle_driver_relation","Driving_experience","Type_of_vehicle","Owner_of_vehicle","Service_year_of_vehicle","Defect_of_vehicle","Area_accident_occured"
                      ,"Lanes_or_Medians","Road_allignment","Types_of_Junction","Road_surface_type","Type_of_collision","Vehicle_movement","Work_of_casuality","Fitness_of_casuality"]


cate_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(sparse_output=False,drop="first",handle_unknown="ignore"))
])


encoder_categorical_features=["Day_of_week","Age_band_of_driver","Sex_of_driver","Road_surface_conditions","Light_conditions","Weather_conditions","Casualty_class","Sex_of_casualty",
                              "Age_band_of_casualty","Casualty_severity","Pedestrian_movement","Cause_of_accident","Time_hour"]

enoder_cate_pipeline=Pipeline([
    ("encoder",OneHotEncoder(sparse_output=False,drop="first",handle_unknown="ignore"))
])



preprocessor=ColumnTransformer([
    ("numeric_columns",numeric_pipeline,numeric_features),
    ("categorcial_columns",cate_pipeline,Imputer_encoder_categorical_features),
    ("encoder_cate",enoder_cate_pipeline,encoder_categorical_features)
])



# final pipeline
Main_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
    n_estimators=200,
    random_state=42
))
])

# fit on training data (X_train, Y_train)
Main_pipeline.fit(X_train, Y_train)

# predict on test data
y_predict = Main_pipeline.predict(X_test)

print(y_predict)




joblib.dump(Main_pipeline,"final_pipeline.pkl")



print("Accuracy score is: ", accuracy_score(Y_test, y_predict))
print("Confusion matrix is:\n", confusion_matrix(Y_test, y_predict))
print(classification_report(Y_test, y_predict))




