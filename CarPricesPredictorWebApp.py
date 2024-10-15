# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 06:03:02 2024

@author: domingosdeeularia
"""

# Dependencies
import pandas as pd
import joblib as jbl
import gradio as gr
import urllib
import sqlite3

# Repository main url
main_url = 'https://raw.githubusercontent.com/domingosdeeulariadumba/CarPricesPrediction/main'

# File containing the preprocessed dataset
db_url = main_url + '/CarPricesPreprocessedDatabase.db'
urllib.request.urlretrieve(url = db_url, filename = 'CarPricesPreprocessedDatabase.db')

# Reading the preprocessed data a pandas dataframe
conn = sqlite3.connect('CarPricesPreprocessedDatabase.db')
query = 'SELECT * FROM CarPrices'
df_prep = pd.read_sql(query, conn)

# Closing the connection
conn.close()

# Importing the model serialized with joblib
model_url = main_url + '/CarPricesPredictionModel.joblib'     
urllib.request.urlretrieve(url = model_url, filename = 'CarPricesPredictionModel.joblib')
model = jbl.load('CarPricesPredictionModel.joblib')

# Function for predictions
def predict_car_price(year: int, make: str, transmission: str, condition: int,
                      odometer: float, color: str, 
                      interior: str, mmr: float) -> float:
        
    # Dataframe with the inputs from the user    
    input_data = pd.DataFrame(data=[[year, make, transmission, condition, 
                                     odometer, color, interior, mmr]],
                              columns=['year', 'make', 'transmission',
                                       'condition', 'odometer', 'color',
                                       'interior', 'mmr'])
                        
    # One Hot Encoding of the input    
    input_dummies = pd.get_dummies(input_data)
        
    # Attributes used to fit the model    
    model_atributtes = model.named_steps['regressor'].feature_names_in_
        
    # Dataframe of attributes used to fit the model containing 0    
    input_final = pd.DataFrame(0, index=[0], columns=model_atributtes)
        
    # Matching the dummies columns    
    input_final[input_dummies.columns] = input_dummies.values
    input_final = input_final[model_atributtes]

    # Prediction result with two decimal places    
    output = round(model.predict(input_final)[0], 2)
        
    return output

# Setting up the interface of the web app
    # Range values
condition_min = int(df_prep.condition.min())
condition_max = int(df_prep.condition.max())    
    # Choices
unique_makes = list(set(df_prep.make))
unique_transmission = list(set(df_prep.transmission))
unique_color = list(set(df_prep.color))
unique_interior = list(set(df_prep.interior))
    # Default values
mode_color = df_prep.color.value_counts().index[0]
mode_make = df_prep.make.value_counts().index[0]
mode_transmission = df_prep.transmission.value_counts().index[0]
mode_interior = df_prep.interior.value_counts().index[0]
mean_mmr = int(df_prep.mmr.mean())
    # Filling the field values
year = gr.Number(label = 'Year', minimum = 1885)
make = gr.Dropdown(label = 'Make', choices = unique_makes, 
                   value = mode_make)
transmission = gr.Dropdown(label = 'Transmission', choices = unique_transmission,
                           value = mode_transmission)
condition = gr.Slider(label = 'Condition', minimum = condition_min, 
                      maximum = condition_max, step = 1, 
                      interactive = True)
odometer = gr.Number(label = 'Odometer', minimum = 0)
color = gr.Dropdown(label = 'Color', choices = unique_color, 
                    value = mode_color)
interior = gr.Dropdown(label = 'Interior Color', choices = unique_interior,
                       value = mode_interior)
mmr = gr.Number(label = 'Manheim Market Report', value = mean_mmr)

# Outputs
sellingprice = gr.Number(label = 'Selling Price')

# Assigning the Gradio interface labels of the web application
car_prices_predictor = gr.Interface(fn = predict_car_price, 
                       description = 'Hello! 👋🏿 Welcome to this car selling price predictor. Please, fill the fields below accordingly!', 
                       inputs = [year, make, transmission, condition, odometer, color, interior, mmr], 
                       outputs = sellingprice, title = 'Car Selling Price Predictor', 
                       allow_flagging = 'auto', theme = 'soft')

# Launching the web application for making predictions
car_prices_predictor.launch(server_name = '0.0.0.0')
