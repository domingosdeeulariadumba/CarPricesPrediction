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
import os

# Repository main url
main_url = 'https://raw.githubusercontent.com/domingosdeeulariadumba/CarPricesPrediction/main'

# File containing the preprocessed dataset
db_url = main_url + '/CarPricesPreprocessedDatabase.db'
urllib.request.urlretrieve(url = db_url, filename = 'CarPricesPreprocessedDatabase.db')

# Reading the preprocessed data as a pandas DataFrame
conn = sqlite3.connect('CarPricesPreprocessedDatabase.db')

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
        odometer, color.lower(), interior.lower(), mmr
        ]], 
        columns = model.feature_names_in_
        )
    output = round(model.predict(input_data).max(), 2)
        
    return output



# Setting up the interface of the web app     
    # Choices
def query_unique(feature):
    unique = [
        i[0] for i in  
        conn.execute(
            f'''
            SELECT DISTINCT {feature} 
            FROM CarPrices
            '''
            ).fetchall()
        ]
    return unique

unique_makes = query_unique('make')
unique_transmission = list(map(lambda t: t.title(), query_unique('transmission')))
unique_color = list(map(lambda c: c.title(), query_unique('color')))
unique_interior = list(map(lambda i: i.title(), query_unique('interior')))

    # Default values
def query_mode(feature):    
    mode = conn.execute(
        f'''
        SELECT {feature}, COUNT(*) as count
        FROM CarPrices
        GROUP BY {feature}
        ORDER BY count DESC
        LIMIT 1
        '''
        ).fetchone()[0]
    return mode    
    
mode_color = query_mode('color').title()
mode_make = query_mode('make')
mode_transmission = query_mode('transmission').title()
mode_interior = query_mode('interior').title()
mean_mmr = int(conn.execute('SELECT AVG(mmr) FROM CarPrices').fetchone()[0])

# Closing the database connection
conn.close()

    # Filling the field values
age = gr.Number(label = 'Age', minimum = 0)
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
port = int(os.environ.get('PORT', 7860))
car_prices_predictor.launch(server_name = '0.0.0.0', server_port = port)
