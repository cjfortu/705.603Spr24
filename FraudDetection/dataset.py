import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold


class Fraud_Dataset:
    """
    A class to partition the dataset.
    """
    def __init__(self, pdpth, dsplit):
        """
        Ingest the dataset and make initial train/val/test splits.
        
        Parameters:
        pdpth (pathlib Path): The dataset file path
        split (tuple of float): The train/val/test proportions
        """
        df = pd.read_csv(pdpth)
        self.dsplit = dsplit
        Y = df['is_fraud'].to_numpy()
        df = df.drop(columns=['is_fraud'])
        X = df.to_numpy()
        self.X_tv, self.X_test, self.Y_tv, self.Y_test =\
                train_test_split(X, Y, test_size=dsplit[2], stratify=Y)
        self.X_train, self.X_val, self.Y_train, self.Y_val =\
                train_test_split(self.X_tv, self.Y_tv,\
                                 test_size = dsplit[1] / (dsplit[0] + dsplit[2]),\
                                 stratify = self.Y_tv)
    
    
    def get_training_dataset(self):
        """
        Get the training data for the 1-fold CV case
        """
        return (self.X_train, self.Y_train)
        
        
    def get_validation_dataset(self):
        """
        Get the validation data for the 1-fold CV case
        """
        return (self.X_val, self.Y_val)
    
    
    def get_testing_dataset(self):
        """
        Get the testing data
        """
        return (self.X_test, self.Y_test)
    
    
    def get_noval_training_dataset(self):
        """
        Get the training data for the no CV case
        """
        return (self.X_tv, self.Y_tv)
    
    
    def get_nfoldCV_datasets(self):
        """
        Split the non-test data into n-fold sets for cross validation.
        
        Parameters:
        n_splits (int): The desired number of splits
        
        Returns:
        X_trains (list of np.array): The X training datasets
        X_vals (list of np.array): The X validation datasets
        Y_trains (list of np.array): The Y training datasets
        Y_vals (list of np.array): The Y validation datasets
        """
        n_splits = int(self.dsplit[0] / self.dsplit[1]) + 1
        X_trains = []
        Y_trains = []
        X_vals = []
        Y_vals = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        for t_idx, v_idx in skf.split(self.X_tv, self.Y_tv):
            X_trains.append(self.X_tv[t_idx])
            Y_trains.append(self.Y_tv[t_idx])
            X_vals.append(self.X_tv[v_idx])
            Y_vals.append(self.Y_tv[v_idx])
            
        return X_trains, X_vals, Y_trains, Y_vals
       