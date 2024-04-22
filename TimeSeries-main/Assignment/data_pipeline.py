import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import MinMaxScaler


class ETL_Pipeline:
    """
    A class to handle prediction target dates.
    """
    def __init__(self):
        self.today = '2022-12-31'
        self.df = pd.read_csv('./data/CreditCardFraudFourYears.csv')
        self.ushols = holidays.UnitedStates(years =\
                                            [2018, 2019, 2020, 2021, 2022, 2023])
        self.npreds = None
    
    
    def getnpreds(self, targetday):
        """
        Get the number of points to predict.
        
        Uses weekly points.
        
        parameters:
        targetday (str): the target prediction day in YYYY-MM-DD
        
        returns:
        npreds (int): the number of points to predict
        """
        today = pd.Timestamp(self.today)
        targetday = pd.Timestamp(targetday)
        delt = (targetday - today) / np.timedelta64(1, 'W')
        self.npreds = int(np.ceil(delt))
        
        return self.npreds
        

    def rnn_preproc(self, X, feat):
        """
        MinMax scale the data
        
        parameters:
        X (dataframe): The data
        feat (str): The feature to operate on
        
        returns:
        Xsc (dataframe): The scaled data
        scaler (MinMaxScaler): The scaler
        """
        print('scaling data for LSTM...')
        # scale the data
        scaler = MinMaxScaler()
        scaler.fit(X[[feat]])
        Xsc = scaler.transform(X[[feat]])

        return Xsc, scaler

           
    def transform(self):
        """
        Create additional feature and sub-feature, slice columns, and aggregate
        for time series learning
        
        Returns:
        X (DataFrame): The processed transaction data
        """
        print('transforming data...')
        # sort chronologically
        df = self.df.sort_values(by = 'unix_time')
        df = df.reset_index()

        # get transactions near midnight
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date'] + ' ' +\
                                                     df['trans_time'])
        df['trans_time_secs'] = df['trans_date_trans_time'].apply(lambda x:\
                x.hour * 3600 + x.minute * 60 + x.second)
        df['near_midnight'] = 0
        df.loc[df['trans_time_secs'].between(0, 14780, inclusive='left') |\
               df['trans_time_secs'].between(78570, 86400, inclusive='right'),\
               'near_midnight'] = 1
        
        # slice columns and resample/aggregate
        df = pd.get_dummies(df, columns = ['is_fraud'])
        df_ts = df[['trans_date_trans_time', 'is_fraud_0', 'is_fraud_1',\
                    'near_midnight']]
        df_ts = df_ts.resample('W', on='trans_date_trans_time').sum()
        
        # add holiday feature
        df_ts['holidays'] = 0
        for period in df_ts.index:
            for hol in self.ushols.keys():
                delt = (period - pd.Timestamp(hol)) / np.timedelta64(1, 'D')
                # A holiday immediately before or after a week bucket could...
                # ...impact the bucket. Hence we expand the window to +-2 days...
                # ...from the week bucket.
                if -1 <= delt <= 9:
                    df_ts.loc[period, 'holidays'] += 1

        # add total transactions and rename/reorder columns
        df_ts['tot_trans'] = df_ts['is_fraud_0'] + df_ts['is_fraud_1']
        df_ts = df_ts.rename(columns={'is_fraud_1': 'fraud_trans'})  # excl is_fraud_0
        df_ts = df_ts[['tot_trans', 'fraud_trans', 'holidays', 'near_midnight']]

        X = df_ts.iloc[:]
        
        return X