# Importing Libraries

from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

# Defining Prediction Function

def predict_rating(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]
    
# Loading Model

model = load_model('car_random_forest_regressor')

# Writing App Title and Description

st.title('Car Star Rating Regressor Web App')
st.write('This is a web app to predict the car star rating of the road based on\
        several features that you can see in the sidebar. Please adjust the\
        value of each feature. After that, click on the Predict button at the bottom to\
        see the prediction of the regressor.')

# Making Sliders and Feature Variables


vehicle_flow         = st.sidebar.slider(label = 'vehicle_flow', min_value = 0.0,
                        max_value = 182124.0 ,
                        value = 11516.0,
                        step = 1000.0)
                        
motorcycle_percent = st.sidebar.slider(label = 'motorcycle_percent', min_value = 3.0,
                        max_value = 8.0 ,
                        value = 7.0,
                        step = 1.0)

ped_peak_hour_flow_across = st.sidebar.slider(label = 'ped_peak_hour_flow_across', min_value = 1.0,
                        max_value = 8.0 ,
                        value = 3.0,
                        step = 1.0)

ped_peak_hour_flow_along_driver_side = st.sidebar.slider(label = 'ped_peak_hour_flow_along_driver_side', min_value = 1.0,
                        max_value = 8.0 ,
                        value = 3.0,
                        step = 1.0)
                        
...
...


# Mapping Feature Labels with Slider Values

features = {
  'vehicle_flow':vehicle_flow,
  'motorcycle_percent':motorcycle_percent,
  'ped_peak_hour_flow_across':ped_peak_hour_flow_across,
  'ped_peak_hour_flow_along_driver_side':	ped_peak_hour_flow_along_driver_side,
  'ped_peak_hour_flow_along_passenger_side':	ped_peak_hour_flow_along_passenger_side,
  'bicycle_peak_hour_flow':	bicycle_peak_hour_flow,
   '''
   '''
   
# Converting Features into DataFrame

features_df  = pd.DataFrame([features])

st.table(features_df)

# Predicting Star Rating

if st.button('Predict'):
    
    prediction = predict_rating(model, features_df)
    
    st.write(' Based on feature values, the car star rating is '+ str(int(prediction)))
    

