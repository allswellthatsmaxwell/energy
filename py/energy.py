import pandas as pd, numpy as np
from os.path import join
import pickle
from enum import Enum
from typing import List, Dict

import keras.backend as K

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
pd.options.mode.chained_assignment = None

DATA_DIR = '../data'
INTERMEDIATE_DIR = '../intermediate'
combined_df_dtypes = {
    'building_id': str,
    'meter': str,
    'timestamp': str,
    'meter_reading': float,
    'site_id': str,
    'primary_use': str,
    'square_feet': int,
    'year_built': int,
    'floor_count': int,
    'air_temperature': float,
    'cloud_coverage': float,
    'dew_temperature': float,
    'precip_depth_1_hr': float,
    'sea_level_pressure': float,
    'wind_direction': float,
    'wind_speed': float,
    'minute': int,
    'second': int,
    'week': int,
    'day': int
}

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
            self.max_rows = max_rows
            self.train_file = join(INTERMEDIATE_DIR, "combined_train.csv")
            self.val_file = join(INTERMEDIATE_DIR, "combined_val.csv")

        def read_train(self):
            try:
                return self.train
            except AttributeError:
                self.train = pd.read_csv(
                    self.train_file, nrows=self.max_rows)
                return self.train

        def read_val(self):
            try:
                return self.val
            except AttributeError:
                self.val = pd.read_csv(
                    self.val_file, nrows=self.max_rows)
                return self.val
            

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

class ModelingPaths:
    scaler = join(INTERMEDIATE_DIR, 'scaler.pkl')
    label_encoder = join(INTERMEDIATE_DIR, 'label_encoder.pkl')
    trained_model = join(INTERMEDIATE_DIR, 'model.keras')


class ModelingConfig:
    def __init__(
            self, paths: ModelingPaths, missing_strategy: MissingStrategy):
        self.paths = paths
        self.missing_strategy = missing_strategy

def load_data_processor(item_name: str, path: str, is_train: bool):
    try:
        item = pickle.load(open(path, 'rb'))
    except FileNotFoundError:
        if not is_train:
            raise_not_train_exception(item=item_name, path=path)
        else:
            return None

class ModelingConfigA(ModelingConfig):
    
    def encode_labels(self, df: pd.DataFrame, is_train: bool):
        le = load_data_processor(
            "Label encoder", self.paths.label_encoder, is_train)
        if le is None:
            le = preprocessing.LabelEncoder()
            le.fit(list(df.loc[:, 'primary_use'].unique()))
            pickle.dump(le, open(self.paths.label_encoder, 'wb'))
        df.loc[:, 'primary_use'] = le.transform(df['primary_use'])
        return df

    def scale_columns(self, X, is_train: bool):
        
        scaler = load_data_processor(
            "Column scaler", self.paths.scaler, is_train)
        if scaler is None:
            scaler = preprocessing.StandardScaler()
            scaler = scaler.fit(X)
            pickle.dump(scaler, open(self.paths.scaler, 'wb'))
        X_scaled = scaler.transform(X)
        return X_scaled

def extract_time_columns_from_timestamps(df) -> None:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['minute'] = df['timestamp'].apply(lambda ts: ts.minute)
    df['second'] = df['timestamp'].apply(lambda ts: ts.second)
    df['week']   = df['timestamp'].apply(lambda ts: ts.week)
    df['day']    = df['timestamp'].apply(lambda ts: ts.day)
    
def prepare_data(df, is_train, predictor_cols: List[str], response_col: str,
                 config: ModelingConfig):
    X_df = df[predictor_cols]
    y = df[response_col].values
    X_df = config.encode_labels(X_df, is_train=is_train)
    X = X_df.values
    X, y = handle_missing_predictor_values(
        X, y, missing_strategy=config.missing_strategy)
    assert not np.isnan(X).any()
    X_scaled = config.scale_columns(X, is_train=is_train)
    return X_scaled, y

def raise_not_train_exception(item, path):
    raise ValueError(
        f"{item} not found at {path}, and is_train=False. " +
        "Will only make an encoder from training data.")
