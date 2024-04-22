import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import pandas as pd
import numpy as np
from pathlib import Path

from data_pipeline import NSTEL_Pipeline
from metrics import Metrics
import tensorflow.keras as tfk
import tensorflow as tf


class Rating_Prediction_Model:
    def __init__(self):
        self.pipeline = NSTEL_Pipeline()
        _, _, self.X_test, _, _, self.Y_test = self.pipeline.loadbatch()
        self.seq = None
        self.model = tfk.models.load_model('./models/text_model.h5')
        
        
    def predsingle(self, text):
        """
        Test the model on a single review.
        
        parameters:
        text (str): The raw review
        
        returns:
        y_pred (int): The predicted star rating
        """
        self.seq = self.pipeline.loadsingle(text)
        self.seq = np.reshape(self.seq, (-1, self.seq.size))
        y_pred = self.model.predict(self.seq)
        self.metrics = Metrics('single', None, y_pred)
        self.metrics.generate_report()
        y_pred = np.argmax(y_pred) + 1
        
        return y_pred
        
    def predbatch(self):
        """
        Test the model.
        
        returns:
        testsz (int): The number of test data reviews
        accuracy (float): The prediction accuracy
        proximalperf (float): The prediction proximal performance (see Metrics)
        """
        Y_preds= self.model.predict(self.X_test)
        self.metrics = Metrics('batch', self.Y_test, Y_preds)
        accuracy, proximalperf = self.metrics.generate_report()
        testsz = Y_preds.shape[0]
        
        return testsz, accuracy, proximalperf
