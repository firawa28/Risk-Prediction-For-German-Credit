# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:06:04 2021

@author: mitra2
"""

# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('logistic_regression_model.pkl','rb'))

@app.route('/api', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([np.array([data['Age'], data['Saving accounts'], data['Checking account'], data['Credit amount'], data['Duration'], data['Sex_female'], data['Sex_male'], data['Job_0'], data['Job_1'], data['Job_2'], data['Job_3'], data['Housing_free'], data['Housing_own'], data['Housing_rent'], data['Purpose_business'], data['Purpose_car'], data['Purpose_domestic appliances'], data['Purpose_education'], data['Purpose_furniture/equipment'], data['Purpose_radio/TV'], data['Purpose_repairs'], data['Purpose_vacation/others']])])

    # Take the first value of prediction
    output = int(prediction[0])
    
    if output == 1:
        hasil = "Good Loan"
    else:
        hasil = "Bad Loan"
            

    return hasil

if __name__ == '__main__':
    app.run(port=5000, debug=True)