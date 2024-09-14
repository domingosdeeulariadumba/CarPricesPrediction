# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 06:03:02 2024

@author: domingosdeeularia
"""

# Dependencies

import pandas as pd
import joblib as jbl
import gradio as gr



# File containing the preprocessed dataset

path =  'https://raw.githubusercontent.com/domingosdeeulariadumba/CarPricesPrediction/main'

file = '/car_prices_train.csv'

df_prep = pd.read_csv(path + file)



# Function for predictions

def predict_car_price(age: int, make: str, transmission: str, condition: int, odometer: float, 
             color: str, interior: str, mmr: float) -> float:

    # Dataframe witth the inputs from user
    
    input_data = pd.DataFrame(data=[[age, make, transmission, condition, odometer, color, interior, mmr]],
                              columns=['age', 'make', 'transmission', 'condition', 'odometer', 'color', 'interior', 'mmr'])

    
    # One Hot Encoding of the input
    
    input_dummies = pd.get_dummies(input_data)

    
    # Importing the model serialized with joblib
    
    model = jbl.load(path + 'car_prices_ml.joblib')

    
    # Atributtes used to fit the model
    
    model_atributtes = model.feature_names_in_

    
    # Dataframe of atributtes used to fit the model containig 0
    
    input_final = pd.DataFrame(0, index = [0], columns = model_atributtes)
    
    
    # Matching the dummies columns
    
    input_final[input_dummies.columns] = input_dummies.values
    input_final = input_final[model_atributtes]

    # Prediction result with two decimal places
    output = round(model.predict(input_final)[0], 2)
    
    return output


# Setting up the interface of the web app

    ## Inputs

age = gr.Number(label = 'Age', minimum = df_prep.age.min(), maximum = df_prep.age.max(),)
make = gr.Dropdown(label = 'Make', choices = list(set(df_prep.make)))
transmission = gr.Dropdown(label = 'Transmission', choices = list(set(df_prep.transmission))) 
condition = gr.Slider(label = 'Condition', minimum = df_prep.condition.min(), maximum = df_prep.condition.max(), step = 1, interactive = True) 
odometer = gr.Number(label = 'Odometer')
color = gr.Dropdown(label = 'Color', choices = list(set(df_prep.color)))
interior = gr.Dropdown(label = 'Interior Color', choices = list(set(df_prep.interior)))
mmr = gr.Number(label = 'Manheim Market Report')

    ## Outputs

sellingprice = gr.Number(label = 'Selling Price')


# Assigning the gradio interface labels of the web application

# Assigning the gradio interface labels of the web application

car_prices_predictor = gr.Interface(fn = predict_car_price, 
                       description = 'Welcome to this price selling predictor. Please, fill the fields below accordingly!', 
                       inputs = [age, make, transmission, condition, odometer, color, interior, mmr], 
                       outputs = sellingprice, title = 'Car Selling Price Predictor', allow_flagging = 'auto', 
                       theme = 'soft')


# Launching the web application for making predictions

car_prices_predictor.launch(server_port=7860)
