# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:06:04 2021

@author: mitra2
"""

# Import libraries
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import json
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('logistic_regression_model.pkl','rb'))

@app.route('/api', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    json_data = json.dumps(data)
    df = pd.read_json(json_data)
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(df)
    
    df['prediction'] = prediction
    
    json_records = df.to_json(orient ='records') 
    
    return json_records

if __name__ == '__main__':
    app.run(port=5000, debug=True)