from flask import Flask, render_template, request
import os

from data_pipeline import ETL_Pipeline
from model import Fraud_Predictor_Model

app = Flask(__name__)

# http://localhost:8786/predict?date=2023-01-24
@app.route('/predict', methods=['GET'])
def getInfer():
    args = request.args
    date = str(args.get('date'))
    npreds = datpipe.getnpreds(date)
    Xtot, sctot = datpipe.rnn_preproc(X, 'tot_trans')
    Xfr, scfr = datpipe.rnn_preproc(X, 'fraud_trans')
    Y_pred_tot, Y_pred_fraud = fpm.predtrans_tot_fraud(Xtot, Xfr, sctot, scfr, npreds)
    return render_template('outputtemp.html', npreds=npreds,
                          Y_pred_tot=Y_pred_tot, Y_pred_fraud=Y_pred_fraud)
    

if __name__ == "__main__":
    flaskPort = 8786
    datpipe = ETL_Pipeline()
    X = datpipe.transform()
    fpm = Fraud_Predictor_Model()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)
