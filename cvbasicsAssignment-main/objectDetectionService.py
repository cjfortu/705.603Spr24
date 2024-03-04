from flask import Flask
from flask import request
import os
from pathlib import Path

from GraphicDataProcessing import ObjectDetection

app = Flask(__name__)

# Use postman to generate the post with a graphic of your choice

@app.route('/post', methods=['POST'])
def get_input():
    imagefile = request.files.get('imagefile', '')
    print('file: ', imagefile)
    imagefile.save('./Pictures/recimg')
    return 'file received'


@app.route('/predict', methods=['GET'])
def get_pred():
    return str(od.procsingle())


if __name__ == "__main__":
    flaskPort = 8786
    od = ObjectDetection()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)
