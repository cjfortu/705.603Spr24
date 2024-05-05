from flask import Flask, render_template, request
import os
import csv

from Data_Pipeline import Point_Pipeline
from Environment import Emailenv
from Agent import Email_Marketing_Agent

app = Flask(__name__)


@app.route('/performance', methods=['GET'])
def get_perf():
    reward, conversion, responses, perftime = agent.evaluate()

    return render_template('evaltemp.html', reward=reward, conversion=conversion, runtime=perftime)


@app.route('/post', methods=['POST'])
def get_input():
    txtfile = request.files.get('textfile')
    txtfile.save(csvpth)
    print('textfile: {} received'.format(txtfile.filename))
    
    return 'textfile: {} received'.format(txtfile.filename)


@app.route('/predict', methods=['GET'])
def get_pred():
    with open(csvpth, newline='') as f:
        reader = csv.reader(f)
        statedat = list(reader)[0]
    print(statedat)
    utypidx = pipeline.proc_usr(statedat[:-1])
    wdfrihol = pipeline.proc_date(statedat[-1])
    stateidx = utypidx + (wdfrihol * 2**18)
    action, conversion = agent.procsingle(stateidx, utypidx, wdfrihol)
    
    return render_template('singletemp.html', gender=statedat[0], age=statedat[1],\
                           custtype=statedat[2], email=statedat[3], tenure=statedat[4],\
                           date=statedat[5], action=action, conversion=conversion)


if __name__ == "__main__":        
    pipeline = Point_Pipeline()
    ubseq, dfres = pipeline.loaddat()
    env = Emailenv(ubseq, dfres, 5)
    agent = Email_Marketing_Agent(env)
    csvpth = './data/statedat.csv'
    flaskPort = 8786
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)
