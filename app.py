# from crypt import methods
from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load('CO2_Emission_regression_model.h5')

app = Flask(__name__)

app.static_folder = 'static'

@app.route('/')
def home():
    return render_template('CO2-Prediction.html')

@app.route('/result', methods=['POST', 'GET'])
def result(): 
    data1 = float(request.form['co2'])
    data1 = np.array([data1]).reshape(-1,1)
    pred = float(model.predict(data1))
    return render_template('Result.html', pred=pred)

if __name__== "__main__":
    app.run(debug=True, port=500)