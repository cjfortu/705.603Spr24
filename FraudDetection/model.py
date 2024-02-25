import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
import sklearn
import json

from data_pipeline import ETL_Pipeline
from dataset import Fraud_Dataset
from metrics import Metrics
from sklearn.metrics import f1_score, roc_auc_score


class Fraud_Detector_Model:
    def __init__(self, datsplit = (0.72, 0.18, 0.1)):
        # Extract, transform, and load data
        print('extracting, transforming, and loading data...')
        self.etlpipe = ETL_Pipeline()
        fpth = Path('./data/transactions-1.csv')
        self.etlpipe.extract(fpth)
        self.etlpipe.transform()
        pdpth = Path('./data/procdata.csv')
        self.etlpipe.load(pdpth)
        
        print('splitting dataset: train={} val={} test={}....'.\
              format(datsplit[0], datsplit[1], datsplit[2]))
        # 10% for testing.  The remaining is split into 5 folds for CV.
        datsets = Fraud_Dataset(pdpth, (0.72, 0.18, 0.1))
        self.X_test, self.Y_test = datsets.get_testing_dataset()
        self.X_train, self.Y_train = datsets.get_noval_training_dataset()
        self.clf = XGBClassifier(tree_method='hist',\
                             device='cuda',\
                             max_depth=8,\
                             objective='binary:logistic')
        print('XGBoost Classifier model ready for training')
        
        self.reppth = Path('./results/report.txt')
        self.metrics = None

        
    def train(self):
        """
        Train the model. 
        """
        self.clf.fit(self.X_train, self.Y_train)
        Y_pred_train = self.clf.predict(self.X_train)
        F1_train = f1_score(self.Y_train, Y_pred_train)
        ROCAUC_train = roc_auc_score(self.Y_train, Y_pred_train)
        print('training performance:\nF1: {}, ROC-AUC score: {}'.\
              format(F1_train, ROCAUC_train))

        
    def test(self):
        """
        Test the model. 
        """
        Y_pred_test = self.clf.predict(self.X_test)
        self.metrics = Metrics(self.Y_test, Y_pred_test)
        self.metrics.generate_report(self.reppth)
    
    
    def get_report(self):
        """
        Write the testing performance report to file and print to both console and web.
        
        Returns:
        outputstr (str): The testing performance web output.
        """
        outputstr = self.metrics.generate_report(self.reppth)
        
        return outputstr
        

    def predict(self):
        """
        Make a prediction for a single data point.
        
        Returns:
        output (str): The web output for prediction/inference.
        """
        df = pd.read_json('./data/recfile.json', orient='records')
        output = ''
        output += 'PREDICT FOR:\n{}'.format(df.loc[0, :].values)
        x = self.etlpipe.proc_single_point(df)    
        y_pred = self.clf.predict(x)
        output += '\nRESULT:'
        if y_pred[0] == 0:
            output += 'not fraud'
        else:
            output += 'is fraud'

        return output
    