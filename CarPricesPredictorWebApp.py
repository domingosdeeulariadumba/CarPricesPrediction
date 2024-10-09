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


# File containing the preprocessed dataset

path = 'https://raw.githubusercontent.com/domingosdeeulariadumba/CarPricesPrediction/main'
file = '/car_prices_train.csv'
df_prep = pd.read_csv(path + file)


# Importing the model serialized with joblib

model_path = (path + '/car_prices_ml.joblib')     
urllib.request.urlretrieve(url=model_path, filename='car_prices_ml.joblib')
model = jbl.load('car_prices_ml.joblib')


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
    
    model_atributtes = model.feature_names_in_
    
    
    # Dataframe of attributes used to fit the model containing 0
    
    input_final = pd.DataFrame(0, index=[0], columns=model_atributtes)
    
    
    # Matching the dummies columns
    
    input_final[input_dummies.columns] = input_dummies.values
    input_final = input_final[model_atributtes]
    input_final_array = input_final.values


    # Prediction result with two decimal places
    
    output = round(model.predict(input_final_array)[0], 2)
    
    
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
mean_mmr = df_prep.mmr.mean()

    # Filling the field values
year = gr.Number(label = 'Year', minimum = 1885,
                 info = 'Enter the manufacturing year of the car. (e.g., 2020)')
make = gr.Dropdown(label = 'Make', choices = unique_makes, 
                   value = mode_make,
                   info = 'Select the manufacturer of the car (e.g., Ford, Toyota).')
transmission = gr.Dropdown(label = 'Transmission', choices = unique_transmission,
                           value = mode_transmission,
                           info = 'Choose the type of transmission (e.g., Automatic, Manual).')
condition = gr.Slider(label = 'Condition', minimum = condition_min, 
                      maximum = condition_max, step = 1, interactive = True,
                      info = 'Select the condition of the car (1 = Poor, 49 = Excellent).')

odometer = gr.Number(label = 'Odometer', minimum = 0, 
                     info = 'Enter the total kilometers driven by the car.')
color = gr.Dropdown(label = 'Color', choices = unique_color, 
                    value = mode_color, 
                    info = 'Select the color of the car.')
interior = gr.Dropdown(label = 'Interior Color', choices = unique_interior,
                       value = mode_interior,
                       info = 'Select the color of the car’s interior.')
mmr = gr.Number(label = 'Manheim Market Report', value = mean_mmr,
                info = 'Enter the market report value for the car.')


# Outputs

sellingprice = gr.Number(label='Selling Price')


# Assigning the Gradio interface labels of the web application

car_prices_predictor = gr.Interface(fn = predict_car_price, 
                       description = 'Hello! 👋🏿 \nWelcome to this car selling price predictor. Please, fill the fields below accordingly!', 
                       inputs = [year, make, transmission, condition, odometer, color, interior, mmr], 
                       outputs = sellingprice, title = 'Car Selling Price Predictor', 
                       allow_flagging = 'auto', theme = 'soft')


# Launching the web application for making predictions

car_prices_predictor.launch()
