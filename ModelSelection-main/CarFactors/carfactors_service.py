from flask import Flask
from flask import request
import os

from carsfactors import carsfactors

app = Flask(__name__)

# http://localhost:8786/infer?transmission=automatic&color=blue&odometer=12000&year=2020&bodytype=suv&price=20000

@app.route('/stats', methods=['GET'])
def getStats():
    return str(cf.model_stats())

@app.route('/infer', methods=['GET'])
def getInfer():
    args = request.args
    manufacturer = args.get('manufacturer')
    transmission = args.get('transmission')
    color = args.get('color')
    odometer = int(args.get('odometer'))
    year = int(args.get('year'))
    engine_type = args.get('engine_type')
    engine_capacity = float(args.get('engine_capacity'))
    bodytype = args.get('bodytype')
    warranty = args.get('warranty')
    drivetrain = args.get('drivetrain')
    price = int(args.get('price'))
    numphotos = int(args.get('numphotos'))
    return cf.model_infer(manufacturer, transmission, color, odometer, year,\
                    engine_type, engine_capacity, bodytype, warranty, drivetrain,\
                    price, numphotos)

@app.route('/post', methods=['POST'])
def hellopost():
    args = request.args
    name = args.get('name')
    location = args.get('location')
    print("Name: ", name, " Location: ", location)
    imagefile = request.files.get('imagefile', '')
    print("Image: ", imagefile.filename)
    imagefile.save('./image.jpg')
    return 'File Received - Thank you'

if __name__ == "__main__":
    flaskPort = 8785
    cf = carsfactors()
    print('starting server...')
    app.run(debug = True, host = '0.0.0.0', port = flaskPort)

