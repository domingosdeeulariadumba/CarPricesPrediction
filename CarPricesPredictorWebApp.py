# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 06:03:02 2024

@author: domingosdeeularia
"""

# Dependencies
import pandas as pd
import joblib as jbl
import gradio as gr
import sqlite3
import urllib

# Repository main url
main_url = 'https://raw.githubusercontent.com/domingosdeeulariadumba/CarPricesPrediction/main'

# File containing the preprocessed dataset
db_url = main_url + '/CarPricesPreprocessedDatabase.db'
urllib.request.urlretrieve(url = db_url, filename = 'CarPricesPreprocessedDatabase.db')

# Reading the preprocessed data as a pandas DataFrame
conn = sqlite3.connect('CarPricesPreprocessedDatabase.db')
query = 'SELECT * FROM CarPrices'
df_prep = pd.read_sql(query, conn)

# Closing the connection
conn.close()

# Importing the serialized model
model_url = main_url + '/CarPricesPredictionModel.joblib'     
urllib.request.urlretrieve(url = model_url, filename = 'CarPricesPredictionModel.joblib')
model = jbl.load('CarPricesPredictionModel.joblib')


# Function for predictions
def predict_price(
        age: int, make: str, transmission: str,  condition: int,
        odometer: float, color: str,  interior: str, mmr: float
       ) -> float:
    
    # Mapping the condition score input to its corresponding class
    scores = range(1, 6)
    classes = ['poor', 'rough', 'average', 'good', 'excellent']
    dict_condition = dict(zip(scores, classes))
    condition = dict_condition[condition]
    
    # Passing input data as a Pandas DataFrame and prediction    
    input_data = pd.DataFrame([[
        age, make, transmission.lower(), condition, 
        odometer, color.title(), interior.title(), mmr
        ]], 
        columns = model.feature_names_in_
        )
    output = round(model.predict(input_data).max(), 2)
        
    return output



# Setting up the interface of the web app
     
    # Choices
unique_makes = list(set(df_prep.make))
unique_transmission = [i.title() for i in set(df_prep.transmission)]
unique_color =[i.title() for i in set(df_prep.color)]
unique_interior = [i.title() for i in set(df_prep.interior)]

    # Default values
mode_color = df_prep.color.mode()[0].title()
mode_make = df_prep.make.mode()[0]
mode_transmission = df_prep.transmission.mode()[0].title()
mode_interior = df_prep.interior.mode()[0].title()
mean_mmr = int(df_prep.mmr.mean())

    # Filling the field values
age = gr.Number(label = 'Year', minimum = 0)
make = gr.Dropdown(label = 'Make', choices = unique_makes, value = mode_make)
transmission = gr.Dropdown(
    label = 'Transmission', 
    choices = unique_transmission,
    value = mode_transmission
    )
condition = gr.Slider(
    label = 'Condition', 
    minimum = 1,
    maximum = 5, 
    step = 1,
    interactive = True
    )
odometer = gr.Number(label = 'Odometer', minimum = 0)
color = gr.Dropdown(label = 'Color', choices = unique_color, value = mode_color)
interior = gr.Dropdown(
    label = 'Interior Color', 
    choices = unique_interior,
    value = mode_interior
    )
mmr = gr.Number(label = 'Manheim Market Report', value = mean_mmr)

# Output field
sellingprice = gr.Number(label = 'Selling Price')

# Launching the web application for making predictions
description = 'Hello! 👋🏿 Welcome to this car selling price predictor. Please,' \
                ' fill the fields below accordingly!'
car_prices_predictor = gr.Interface(fn = predict_price, 
                       description = description,
                       inputs = [
                           age, make, transmission, condition, 
                           odometer, color, interior, mmr
                           ], 
                       outputs = sellingprice, 
                       title = 'Car Selling Price Predictor', 
                       allow_flagging = 'auto', theme = 'soft')
car_prices_predictor.launch(server_name = '0.0.0.0')