from flask import Flask
from flask import request
import os

from model import Fraud_Detector_Model

app = Flask(__name__)

# http://localhost:8786/performance
# http://localhost:8786/predict


@app.route('/performance', methods=['GET'])
def get_perf():
    return str(fdm.get_report())


@app.route('/post', methods=['POST'])
def get_input():
    jsonfile = request.files.get('jsonfile')
    print('file: ', jsonfile.filename)
    jsonfile.save('./data/recfile.json')
    return 'file received'


@app.route('/predict', methods=['GET'])
def get_pred():
    return str(fdm.predict())
    

if __name__ == "__main__":
    flaskPort = 8786
    fdm = Fraud_Detector_Model()
    fdm.train()
    fdm.test()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)


