# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:11:51 2021

@author: mitra2
"""

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import json
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('logistic_regression_model.pkl','rb'))

def get_record_dummy(keys, values):    
    dataframe = pd.DataFrame(keys, columns = ['columns'], index = keys)
    dataframe_dummy = pd.get_dummies(dataframe)
    list_value = dataframe_dummy.loc[values].tolist()
    return list_value

def get_label_value(keys, value):
    le = LabelEncoder()
    dataframe = pd.DataFrame(keys, columns = ['columns'], index = keys)
    dataframe['columns'] = le.fit_transform(dataframe['columns'])
    values = int(dataframe.loc[value])
    return values

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data, columns = ['Age', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Sex', 'Job', 'Housing', 'Purpose'])
    
    array_df = []
    
    for subscript in range(0, len(df)):
        sex = get_record_dummy(['male', 'female'], df['Sex'].loc[subscript])
        job = get_record_dummy(['0', '1', '2', '3'], df['Job'].loc[subscript])
        housing = get_record_dummy(['free', 'own', 'rent'], df['Housing'].loc[subscript])
        purpose = get_record_dummy(['business', 'car', 'domestic appliances', 'education', 'furniture/equipment', 'radio/TV', 'repairs', 'vacation/others'], df['Purpose'].loc[subscript])
        saving_accounts = get_label_value(['None', 'little', 'quite rich', 'rich', 'moderate'], df['Saving accounts'].loc[subscript])
        checking_accounts = get_label_value(['little', 'moderate', 'None', 'rich'], df['Checking account'].loc[subscript])
        
        lists = [df['Age'].loc[subscript], saving_accounts, checking_accounts, df['Credit amount'].loc[subscript], df['Duration'].loc[subscript], sex, job, housing, purpose]
        
        arrays = []
        for i in lists:
            arrays = np.append(arrays, i)
        
        array_df.append(arrays)
        
    test_set = pd.DataFrame(array_df, columns = ['Age', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Sex_male', 'Sex_female', 'Job_0', 'Job_1', 'Job_2', 'Job_3', 'Housing_free', 'Housing_own', 'Housing_rent', 'Purpose_business', 'Purpose_car', 'Purpose_domestic appliances', 'Purpose_education', 'Purpose_furniture/equipment', 'Purpose_radio/TV', 'Purpose_repairs', 'Purpose_vacation/others'])
    
    result = model.predict(test_set)
    
    df['prediction'] = result
    
    json_records = df.to_json(orient ='records')
    
    return json_records

if __name__ == '__main__':
    app.run(port=5000, debug=True)
        
        
        