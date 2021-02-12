import pickle
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pandas as pd

f = open('./rf.pkl', 'rb')
cl = pickle.load(f)

app = Flask(__name__)

@app.route('/predict/form', methods=['POST'])
def predict_iris_form():
    s_length = request.form["s_length"]
    s_width = request.form["s_width"]
    p_length = request.form["p_length"]
    p_width = request.form["p_width"]

    prediction = cl.predict(np.array([[s_length, s_width, p_length, p_width]]))
    return str(prediction)

@app.route('/predict/file', methods=['POST'])
def predict_iris_file():
    input_data = pd.read_csv(request.files.get('input_file'), header=None)
    prediction = cl.predict(input_data)
    return str(list(prediction))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)