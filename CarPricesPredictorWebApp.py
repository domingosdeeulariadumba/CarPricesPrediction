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
import os
import flask
import threading

# Initialize Flask app
app = flask.Flask(__name__)

# File containing the preprocessed dataset
path = 'https://raw.githubusercontent.com/domingosdeeulariadumba/CarPricesPrediction/main'
file = '/car_prices_train.csv'
df_prep = pd.read_csv(path + file)

# Importing the model serialized with joblib
model_path = (path + '/car_prices_ml.joblib')     
urllib.request.urlretrieve(url = model_path, filename = 'car_prices_ml.joblib')
model = jbl.load('car_prices_ml.joblib')

# Function for predictions
def predict_car_price(year: int, make: str, transmission: str, condition: int,
                      odometer: float, color: str, 
                      interior: str, mmr: float) -> float:
    
    # Dataframe with the inputs from the user
    input_data = pd.DataFrame(data = [[year, make, transmission, condition, 
                                     odometer, color, interior, mmr]],
                              columns = ['year', 'make', 'transmission',
                                       'condition', 'odometer', 'color',
                                       'interior', 'mmr'])
    
    # One Hot Encoding of the input
    input_dummies = pd.get_dummies(input_data)
    
    # Attributes used to fit the model
    model_atributtes = model.feature_names_in_
    
    # Dataframe of attributes used to fit the model containing 0
    input_final = pd.DataFrame(0, index = [0], columns = model_atributtes)
    
    # Matching the dummies columns
    input_final[input_dummies.columns] = input_dummies.values
    input_final = input_final[model_atributtes]
    input_final_array = input_final.values

    # Prediction result with two decimal places
    output = round(model.predict(input_final_array)[0], 2)
    
    return output

# Setting up the interface of the web app
condition_min = int(df_prep.condition.min())
condition_max = int(df_prep.condition.max())
unique_makes = list(set(df_prep.make))
unique_transmission = list(set(df_prep.transmission))
unique_color = list(set(df_prep.color))
unique_interior = list(set(df_prep.interior))

year = gr.Number(label = 'Year', minimum = 1885)
make = gr.Dropdown(label = 'Make', choices = unique_makes)
transmission = gr.Dropdown(label ='Transmission', choices = unique_transmission)
condition = gr.Slider(label='Condition', minimum = condition_min, 
                      maximum = condition_max, step = 1, interactive = True)
odometer = gr.Number(label = 'Odometer')
color = gr.Dropdown(label = 'Color', choices = unique_color)
interior = gr.Dropdown(label = 'Interior Color', choices = unique_interior)
mmr = gr.Number(label = 'Manheim Market Report')

# Outputs
sellingprice = gr.Number(label = 'Selling Price')

# Specifying the directory for flagged data
flagging_dir = 'flags'

# Ensuring the flagging directory exists
os.makedirs(flagging_dir, exist_ok = True)

# Flask endpoint for flagged data
@app.route('/flags', methods = ['GET'])
def get_flags():
    try:
        flags = os.listdir(flagging_dir)  # Listing all files in the flags directory
        return flask.jsonify(flags)  # Returning the list of files as JSON
    except Exception as e:
        return flask.jsonify({"error": str(e)}), 500
    
# Flask endpoint to serve a specific flagged file
@app.route('/flags/<filename>', methods = ['GET'])
def serve_flagged_file(filename):
    try:
        return flask.send_from_directory(flagging_dir, filename)
    except Exception as e:
        return flask.jsonify({"error": str(e)}), 500

# Assigning the Gradio interface labels of the web application
car_prices_predictor = gr.Interface(fn=predict_car_price, 
                                     description='Welcome to this car selling price predictor. Please, fill the fields below accordingly!', 
                                     inputs=[year, make, transmission, condition, odometer, color, interior, mmr], 
                                     outputs=sellingprice, title='Car Selling Price Predictor', 
                                     flagging_mode = 'auto', flagging_dir=flagging_dir,
                                     theme='soft')

# Function to launch Gradio in a separate thread
def launch_gradio():
    car_prices_predictor.launch(server_name = '0.0.0.0')

# Launching both Flask and Gradio
if __name__ == '__main__':
    
    # Starting Gradio App in a separate thread
    threading.Thread(target = launch_gradio).start()
    
    # Start the Flask app
    app.run(host = '0.0.0.0', port = 8080)
