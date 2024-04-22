from flask import Flask, render_template, request
import os
import numpy as np

from data_pipeline import Pipeline
from model import Plate_Detection_Model

app = Flask(__name__)


@app.route('/performance', methods=['GET'])
def get_perf():
    miou, novboxes, nodet, smmets, smnbnetmets, smperfrat, smnbperfrat = pdm.performance_test()

    return render_template('testmettemp.html', miou = miou, novboxes = novboxes,\
                           nodet = nodet, smmets = smmets, smnbnetmets = smnbnetmets,\
                           smperfrat = smperfrat, smnbperfrat = smnbperfrat)


@app.route('/predictvid', methods=['GET'])
def get_pred():
    fullarr = np.load('./data/trimarr.npy')
    platecnt, detplatecnt = pdm.predict(fullarr)

    return render_template('vidpredtemp.html', platecnt=platecnt, detplatecnt=detplatecnt)
    

if __name__ == "__main__":        
    def setup_pipe():
        pipeline = Pipeline(input_url='udp://127.0.0.1:23000',
                     width=3840, height=2160,
                     trimsize=[[880, 2160], [640, 3200]],
                     viddur=60, dfps=2, npypth='./data/trimarr.npy')
        pipeline.extract()
        pipeline.transform()
        pipeline.load()
        
    setup_pipe()
    
    dsize = (416, 416)
    pdm = Plate_Detection_Model(dsize=dsize, verbose=False, yolover='tiny',\
                                bboxdiag=False, conf=0.3, nms=0.2)
        
    
    flaskPort = 8785
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)


