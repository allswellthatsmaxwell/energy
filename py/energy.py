import pandas as pd, numpy as np
from os.path import join
from enum import Enum

import keras.backend as K
from sklearn.impute import SimpleImputer

DATA_DIR = '../data'
INTERMEDIATE_DIR = '../intermediate'
class Data:    
    class Weather:
        def __init__(self, max_rows=None):
            train_file = join(DATA_DIR, "weather_train.csv")
            test_file = join(DATA_DIR, "weather_test.csv")
            self.train = pd.read_csv(train_file, nrows=max_rows)

    class Meter:
        def __init__(self, max_rows=None):
            train_file = join(DATA_DIR, "train.csv")
            test_file = join(DATA_DIR, "test.csv")
            self.train = pd.read_csv(train_file, nrows=max_rows)

    class Buildings:
        def __init__(self):
            file = join(DATA_DIR, "building_metadata.csv")
            self.data = pd.read_csv(file)

    class Merged:
        def __init__(self, max_rows=None):
            self.train_file = join(INTERMEDIATE_DIR,
                                   "combined_data_16000000.csv")
            self.data = pd.read_csv(self.train_file, nrows=max_rows)

def merge_data(weather_df, meter_df, buildings_df):
    return (
        meter_df.
        merge(buildings_df, on='building_id', how='left').
        merge(weather_df, on=['site_id', 'timestamp']))

def define_rmsle(n_obs: int):
    def rmsle(y_true, y_pred):
        log_difference = (K.log(y_pred + K.constant(1)) -
                          K.log(y_true + K.constant(1)))
        scaling = 1 / n_obs
        raised_log = K.pow(log_difference, 2)
        squared_log_raised = K.square(raised_log)
        sum_of_squared_log_raised = K.sum(K.pow(log_difference, 2))
        scaled_sum_of_squared_log_raised = scaling * sum_of_squared_log_raised
        return K.sqrt(scaled_sum_of_squared_log_raised)
    return rmsle

class MissingStrategy(Enum):
    REMOVE_ROW = 1
    USE_COLUMN_MEAN = 2
    USE_COLUMN_MEDIAN = 3
    PREDICT = 4

def handle_missing_predictor_values(X, y, missing_strategy):
    if missing_strategy == MissingStrategy.REMOVE_ROW:
        nan_rowinds = X_df.apply(lambda row: pd.isnull(row).any(), axis=1)
        X = X[~nan_rowinds]
        y = y[~nan_rowinds]
    elif missing_strategy in (MissingStrategy.USE_COLUMN_MEAN, 
                              MissingStrategy.USE_COLUMN_MEDIAN):
        if missing_strategy == MissingStrategy.USE_COLUMN_MEAN:
            strategy = 'mean'
        else:
            strategy = 'median'
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
        imp = imp.fit(X)  
        X = imp.transform(X)
    else:
        raise NotImplementedError(f"{missing_strategy} not implemented")
    return X, y

def evaluate_model(model, validation_df):
    ## We should make one of those SKLearn pipeline things
    ## to prepare each df the same.
    pass
