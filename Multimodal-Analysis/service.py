from flask import Flask, render_template, request
import os
import numpy as np

from model import Rating_Prediction_Model

app = Flask(__name__)


@app.route('/performance', methods=['GET'])
def get_perf():
    testsz, accuracy, proximalperf = rpm.predbatch()
    
    return render_template('batchtemp.html', testsz=testsz, accuracy=accuracy, proximalperf=proximalperf)


@app.route('/post', methods=['POST'])
def get_input():
    txtfile = request.files.get('textfile')
    txtfile.save(txtpth)
    print('textfile: {} received'.format(txtfile.filename))
    
    return 'textfile: {} received'.format(txtfile.filename)


@app.route('/predict', methods=['GET'])
def get_pred():
    with open(txtpth, 'r') as f:
        contents = f.read()
    y_pred = rpm.predsingle(contents)
    
    return render_template('singletemp.html', contents=contents, rating=y_pred)
    

if __name__ == "__main__":        
    rpm = Rating_Prediction_Model()
    txtpth = './data/textfile.txt'
    flaskPort = 8786
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)
