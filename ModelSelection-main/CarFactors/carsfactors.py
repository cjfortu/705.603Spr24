# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor


class carsfactors:
    def __init__(self):
        self.modelLearn = False
        self.stats = None
        # allow conversion to numerical ordinality according to vehicle size
        self.body_switch = {
            'universal': 0,
            'hatchback': 1,
            'cabriolet': 2,
            'coupe': 3,
            'sedan': 4,
            'liftback': 5,
            'suv': 6,
            'minivan': 7,
            'van': 8,
            'pickup': 9,
            'minibus': 10,
            'limousine': 11
        }
        # ordinal columns which will be normalized
        self.normcols = ['odometer_value', 'year_produced', 'engine_capacity',\
                         'body_type', 'price_usd', 'number_of_photos']
        # store the maximum ordinal values from the training data, to be used for normalization
        self.maxes = {
            'odometer_value': None,
            'year_produced': None,
            'engine_capacity': None,
            'body_type': None,
            'price_usd': None,
            'number_of_photos': None
        }
        # store the columns which will be one hot encoded
        self.dumcols = ['manufacturer_name', 'transmission', 'color', 'engine_type',\
                        'has_warranty', 'drivetrain']
        # store all feature columns
        self.cols = None
        

    def model_learn(self):
        # Importing the dataset into a pandas dataframe
        df = pd.read_csv('cars.csv')
        
        # Remove NaN rows
        df = df.dropna()
        
        # identify manufacturer-model combinations for train_test_split stratification
        df['manumod'] = df[['manufacturer_name', 'model_name']].apply(lambda row:\
                '_'.join(row.values.astype(str)), axis=1)
        
        # convert manufacturer-model combinations to integer value
        manumod_switch = {}
        manumodnum = 0
        for manumod in df['manumod'].value_counts().items():
            if manumod[1] < 2:
                manumod_switch[manumod[0]] = df['manumod'].value_counts().size
            elif manumod[0] in manumod_switch.keys():
                pass
            else:
                manumod_switch[manumod[0]] = manumodnum
                manumodnum += 1
        df['manumod_num'] = df['manumod'].map(manumod_switch)
        test_size = df['manumod_num'].value_counts().size

        #Remove Unwanted Columns
        df = df[['manumod_num', 'manufacturer_name', 'transmission', 'color', 'odometer_value',\
                 'year_produced', 'engine_type', 'engine_capacity', 'body_type',\
                 'has_warranty', 'drivetrain', 'price_usd', 'number_of_photos',\
                 'duration_listed']]
        
        # ordinally encode body_type to reflect that some cars are bigger than others.  
        # This is the order 'universal','hatchback', 'cabriolet','coupe','sedan','liftback', 'suv', 'minivan', 'van','pickup', 'minibus','limousine'
        df = df.replace({'body_type': self.body_switch})
        
        # Feature Scaling - required due to different orders of magnitude across the features
        # make sure to save the scaler for future use in inference
        for col in self.normcols:
            self.maxes[col] = df[col].max()
            df[col] = df[col] / self.maxes[col]
    
    
        # OHE manufacturer_name, transmission, color, engine_type, has_warranty, drivetrain
        df = pd.get_dummies(data=df, columns=self.dumcols)
        
        # Seperate X and y (features and label)  The last feature "duration_listed" is the label
        Y = df['duration_listed'].to_numpy()
        strat_arg = 'manumod_num'
        strat = df[strat_arg].to_numpy()  # stratify according to manufacturer-model combination
        dffeat = df.drop(columns=['manumod_num', 'duration_listed'])
        X = dffeat.to_numpy()
        self.cols = list(dffeat.columns)  # keep the feature names for inference
        
        # set the random forest hyperparameters
        max_features = 0.5  # use %50 of the available features for each tree
        n_estimators = X.shape[1]  # take the same number of estimators as there are features
        min_samples_leaf = 2  # do not allow singleton leaves
        
        print('random forest hyperparameters:')
        print('max_features: {}'.format(max_features))
        print('n_estimators: {}'.format(n_estimators))
        print('min_samples_leaf: {}'.format(min_samples_leaf))
        print('training/testing attributes:')
        print('trainint shape: {}'.format(X.shape))
        print('test_size: {}'.format(test_size))
        print('stratify train_test_split according to manufacturer-model combination: {}'.format(strat))

        # Select useful model to deal with regression (it is not categorical for the number of days can vary quite a bit)
        print('training...')
        # Splitting the dataset into the Training set and Test set 
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,\
                                                           stratify=strat,
                                                           random_state=2)
        # we will use a random forest
        self.model = RandomForestRegressor(n_jobs = -1,
                                          n_estimators = n_estimators,
                                          min_samples_leaf = min_samples_leaf,
                                          max_features = max_features)
        self.model.fit(X_train, Y_train)
        Y_pred_train = self.model.predict(X_train)
        Y_pred_test = self.model.predict(X_test)

        # we will use root mean squared error as the performance metric
        rmse_train = np.sqrt(mse(Y_train, Y_pred_train))
        rmse_test = np.sqrt(mse(Y_test, Y_pred_test))
        
        # keep the original provided metric
        score = self.model.score(X_train, Y_train)
        
        self.stats = [score, rmse_train, rmse_test]
        self.modelLearn = True

    # this demonstrates how you have to convert these values using the encoders and scalers above (if you choose these columns - you are free to choose any you like)
    def model_infer(self, manufacturer, transmission, color, odometer, year,\
                    engine_type, engine_capacity, bodytype, warranty, drivetrain,\
                    price, numphotos):
        if(self.modelLearn == False):
            self.model_learn()

        # initialize zeroed single row dataframe
        df = pd.DataFrame([[0 for i in self.cols]], columns = self.cols)
        
        # input one hot encoded features
        for keyroot, val in zip(self.dumcols,\
                                [manufacturer, transmission, color,\
                                engine_type, warranty, drivetrain]):
            df[(keyroot + '_{}').format(val)] = 1
        
        # convert bodytype according to size ordinality:
        df['body_type'] = self.body_switch[bodytype]
        
        # input normalized ordinal features
        for col, val in zip(self.normcols, [odometer, year, engine_capacity, df.loc[0, 'body_type'],\
                                            price, numphotos]):
            df[col] = val / self.maxes[col]
        
        # convert dataframe to numpy array
        x = df.loc[0].to_numpy().reshape(1, -1)
        
        #determine prediction
        y_pred = self.model.predict(x)
        return str(y_pred)
        
    def model_stats(self):
        if(self.modelLearn == False):
            self.model_learn()
        print('result...\nscore: {}\nRMSE -- train: {}, test: {}'.\
            format(self.stats[0], self.stats[1], self.stats[2]))

        return str(self.stats)
