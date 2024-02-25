import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy
from sklearn.preprocessing import StandardScaler


class ETL_Pipeline:
    """
    A class to extract, transform, and load machine learning data.
    """
    def __init__(self):
        self.df = None
        self.zcsjcols = []
        self.zcsj = ['zip', 'city', 'state', 'job']
        self.valscountsall = []
        self.scaling_model = None
        self.cols2std = ['amt', 'year', 'dob_year']
        self.dropcols = ['Unnamed: 0', 'cc_num', 'merchant', 'first', 'last',\
                              'street', 'trans_num', 'unix_time', 'merch_lat', 'lat',\
                              'merch_long', 'long', 'city_pop']
        self.dummycols = None
        self.featcols = None
    
    
    def extract(self, fpth):
        """
        create a DataFrame from a csv
        
        parameters:
        fpth (pathlib Path): the source file path
        """
        self.df = pd.read_csv(fpth)
    
    
    def facetize_trans_date_trans_time(self, df):
        """
        Sine/cosine facetize day within a year, and seconds within a day
        for trans_date_trans_time
        
        Used by:
        transform()
        proc_single_point()
        
        Parameters:
        df (DataFrame): The transaction data
        
        Returns:
        df (DataFrame): The transaction data
        """
        # convert trans_date_trans_time to datetime objects
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        
        # start process to facetize trans_date_trans_time
        df['year'] = df['trans_date_trans_time'].dt.year
        df['day_of_year'] = df['trans_date_trans_time'].dt.day_of_year
        df['sec_of_day'] = df['trans_date_trans_time'].dt.second +\
                60 * df['trans_date_trans_time'].dt.minute +\
                3600 * df['trans_date_trans_time'].dt.hour
        
        # First do day_of_year
        df['doy_x'] = np.cos(2 * np.pi * df.loc[df['trans_date_trans_time'].dt.is_leap_year,\
                'day_of_year'] / 366)  # x component of day_of_year for leap years
        df.loc[~df['trans_date_trans_time'].dt.is_leap_year, 'doy_x'] =\
                np.cos(2 * np.pi * df.loc[~df['trans_date_trans_time'].dt.is_leap_year,\
                'day_of_year'] / 365)  # x component of day_of_year for non leap years
        df['doy_y'] = np.sin(2 * np.pi * df.loc[df['trans_date_trans_time'].dt.is_leap_year,\
                'day_of_year'] / 366)  # y component of day_of_year for leap years
        df.loc[~df['trans_date_trans_time'].dt.is_leap_year, 'doy_y'] =\
                np.sin(2 * np.pi * df.loc[~df['trans_date_trans_time'].dt.is_leap_year,\
                'day_of_year'] / 365)  # y component of day_of_year for non leap years

        # Now do sec_of_day
        df['sod_x'] = np.cos(2 * np.pi * df['sec_of_day'] / 86400)  # x component
        df['sod_y'] = np.sin(2 * np.pi * df['sec_of_day'] / 86400)  # y component
        
        # drop unnecessary columns
        df = df.drop(columns=['trans_date_trans_time', 'day_of_year', 'sec_of_day'])
        
        return df
        
        
    def groupZCSJ(self):
        """
        Group zip, city, state, and job
        
        Grouping is those with perfect fraud conditional probability, and those with a 
        conditional probability at least an order of magnitude less
        
        Used by:
        transform()
        """
        for param in self.zcsj:
            self.zcsjcols.append('{}_cp1'.format(param))
            valscounts = (self.df.loc[self.df['is_fraud'] == 1, '{}'.format(param)].\
                    value_counts() / self.df['{}'.format(param)].\
                    value_counts()).sort_values(ascending=False) == 1.0
            self.valscountsall.append(valscounts)
    
    
    def oheZCSJ(self, df=None, infer=False):
        """
        OHE zip, city, state, and job
        
        Used by:
        transform()
        proc_single_point()
        
        Parameters:
        df (DataFrame): The transaction data
        infer (bool): Whether for single point inference or not
        
        Returns:
        df (DataFrame): The transaction data
        """
        if infer == False:
            df = deepcopy(self.df)
        else:
            df = df
            
        for zcsjcol, param, valscounts in zip(self.zcsjcols, self.zcsj, self.valscountsall):
            df[zcsjcol] = 0
            truevals = []
            for ival, bval in valscounts.items():
                if bval == True:
                    truevals.append(ival)

            df.loc[df['{}'.format(param)].isin(truevals), zcsjcol] = 1
        df = df.drop(columns=self.zcsj)
            
        # drop unnecessary columns from OHE zip, city, state, job
        for col in df.columns:
            if col[-1] == '0':
                df = df.drop(columns=[col])

        if infer == False:
            self.df = df
            return None
        else:
            return df
    
    
    def facetize_dob(self, df):
        """
        Sine/cosine facetize day within a year, and seconds within a day
        for dob
        
        Used by:
        transform()
        proc_single_point()
        
        Parameters:
        df (DataFrame): The transaction data
        
        Returns:
        df (DataFrame): The transaction data
        """
        df['dob'] = pd.to_datetime(df['dob'])
        
        # start process to facetize dob
        df['dob_year'] = df['dob'].dt.year
        df['dob_doy'] = df['dob'].dt.day_of_year
        
        # now sine/cosine facetize
        df['dob_doy_x'] =\
                np.cos(2 * np.pi * df.loc[df['dob'].dt.is_leap_year, 'dob_doy'] /\
                366)  # x component of day_of_year for leap years
        df.loc[~df['dob'].dt.is_leap_year, 'dob_doy_x'] =\
                np.cos(2 * np.pi * df.loc[~df['dob'].dt.is_leap_year, 'dob_doy'] /\
                365)  # x component of day_of_year for non leap years
        df['dob_doy_y'] =\
                np.sin(2 * np.pi * df.loc[df['dob'].dt.is_leap_year, 'dob_doy'] /\
                366)  # y component of day_of_year for leap years
        df.loc[~df['dob'].dt.is_leap_year, 'dob_doy_y'] =\
                np.sin(2 * np.pi * df.loc[~df['dob'].dt.is_leap_year, 'dob_doy'] /\
                365)  # y component of day_of_year for non leap years
        
        # drop unecessary columns
        df = df.drop(columns=['dob', 'dob_doy'])
        
        return df
    
    
    def get_scaling(self):
        """
        Get the standard scaling for the training data
        
        Used By:
        transform()
        """
        std_scaler = StandardScaler()
        self.scaling_model = std_scaler.fit(self.df[self.cols2std])
        
    
    def normalize_ordinal_dat(self, df=None, infer=False):
        """
        Normalize ordinal data using the standard scaling from training data
        
        Used By:
        transform()
        proc_single_point()
        
        Parameters:
        df (DataFrame): The transaction data
        infer (bool): Whether for single point inference or not
        
        Returns:
        df (DataFrame): The transaction data
        """
        if infer == False:
            df = deepcopy(self.df)
        else:
            df = df
            
        df[self.cols2std] = self.scaling_model.transform(df[self.cols2std])
        
        if infer == False:
            self.df = df
            return None
        else:
            return df
        
    
    def transform(self):
        """
        Clean, process, and prepare the data for modeling.
        
        Uses:
        facetize_trans_date_trans_time()
        groupZCSJ()
        oheZCSJ()
        facetize_dob()
        get_scaling()
        normalize_ordinal_dat()
        """        
        df = deepcopy(self.df)
        # get rid of unneeded columns
        df = df.drop(columns=self.dropcols)
        
        # preservce cyclical nature of day within year and sec within day
        df = self.facetize_trans_date_trans_time(df)
        
        # OHE category and sex
        df = pd.get_dummies(data=df, columns=['category', 'sex'])
        self.dummycols = [col for col in df.columns\
                if 'category' in col or 'sex' in col]
        
        self.df = df
        
        # group and OHE zip, city, state, job
        self.groupZCSJ()
        self.oheZCSJ(infer=False)
        
        # preserve cyclical nature of day within year
        df = deepcopy(self.df)
        self.df = self.facetize_dob(df)
        
        self.get_scaling()
        self.normalize_ordinal_dat(infer=False)
        featcols = list(self.df.columns)
        featcols.remove('is_fraud')
        self.featcols = featcols
        
        
    def load(self, pdpth):
        """
        Write the engineered data to a csv
        
        Parameters:
        pdpth (pathlib Path): the processed data file path
        """
        self.df.to_csv(pdpth, index=False)
        
        
    def proc_single_point(self, df):
        """
        Apply the transformation to a single input,
        while preserving the grouping and scaling from the training data.
        
        Uses:
        facetize_trans_date_trans_time()
        oheZCSJ()
        facetize_dob()
        normalize_ordinal_dat()
        
        Returns:
        x (np.array): a single point numpy array
        """
        df = self.facetize_trans_date_trans_time(df)
        df = self.oheZCSJ(df=df, infer=True)  # using grouping from training data
        df = self.facetize_dob(df)
        df = self.normalize_ordinal_dat(df=df, infer=True)  # use scaling from training data
        
        # handle category and sex using training data columns
        for dummycol in self.dummycols:
            df[dummycol] = 0
        df['category_{}'.format(df.loc[0, 'category'])] = 1
        df['sex_{}'.format(df.loc[0, 'sex'])] = 1      
        df = df.drop(columns = ['category', 'sex'])
        
        df = df.loc[:, self.featcols]  # reorder the columns according to the training data
        
        # convert dataframe to numpy array
        x = df.loc[0].to_numpy().reshape(1, -1)
        
        return x
        