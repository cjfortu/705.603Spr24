import pickle
import numpy as np
# load LSTM packages
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow.keras as tfk

from data_pipeline import ETL_Pipeline


class Fraud_Predictor_Model:
    """
    A class to predict quantity of total and fraudulent transactions.
    """
    def __init__(self):
        self.model = None
        self.X = None
        self.npreds = None
        self.scaler = None
        
        
    def pred_LSTM(self):
        """
        Use the SARIMAX model to predict future values
        
        returns:
        Y_pred_tot (np.array): The predictions
        """

        Y_pred = []
        # the last sequence of inputs leading to edge of prediction
        first_eval_batch = self.X[-52:]
        current_batch = first_eval_batch.reshape((1, 52, 1))
        # predict 1 bucket at a time into the test data range
        for i in range(self.npreds):
            # get next bucket prediction
            current_pred = self.model.predict(current_batch, verbose=0)[0]
            # store prediction
            Y_pred.append(current_pred)
            # update batch to now include prediction and drop first value
            current_batch = np.append(current_batch[:, 1:, :],
                                      np.array([[current_pred]]), axis=1)

        # Reshaping/slice due to the dynamics of Numpy, Tensorflow, and Sklearn
        Y_pred = np.squeeze(self.scaler.inverse_transform(Y_pred), axis=1)
        print('LSTM model prediction complete.\n')

        return Y_pred
              
              
    def predtrans_tot_fraud(self, Xtot, Xfr, sctot, scfr, npreds):
        """
        Make predictions for total transactions and fraudulent transactions
        
        parameters:
        Xtot (DataFrame): The total transaction data
        Xfr (DataFrame): The fraudulent transaction data
        sctot (MinMaxScaler): The scaler for total transactions
        scfr (MinMaxScaler): The scaler for fraudulent transactio
        npreds (int): The number of predictions to make
        
        returns:
        Y_pred_tot (np.array): The total transaction predictions
        Y_pred_fraud (np.array): The fraudulent transaction predictions
        """
        self.npreds = npreds

        self.model = tfk.models.load_model('./modeldat/tot_trans-None_64.tfmod')
        self.X = Xtot
        self.scaler = sctot
        Y_pred_tot = self.pred_LSTM()
        self.model = tfk.models.load_model('./modeldat/fraud_trans-None_64.tfmod')
        self.X = Xfr
        self.scaler = scfr
        Y_pred_fraud = self.pred_LSTM()
              
        return Y_pred_tot, Y_pred_fraud
