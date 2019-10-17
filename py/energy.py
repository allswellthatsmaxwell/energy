import pandas as pd, numpy as np
from os.path import join

import keras.backend as K

DATA_DIR = '../data'
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
        
def merge_data(weather_df, meter_df, buildings_df):
    return (
        meter_df.
        merge(buildings_df, on='building_id', how='left').
        merge(weather_df, on=['site_id', 'timestamp']))

def define_rmsle(n_obs: int):
    def rmsle(y_true, y_pred):
        log_difference = (K.log(y_pred + 1) -
                          K.log(y_true + 1))
        scaling = 1 / n_obs
        return K.sqrt(K.prod(K.constant(scaling),
                            K.sum(log_difference**2)))
    return rmsle
