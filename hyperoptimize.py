#!/usr/bin/env python3

import numpy as np
import pandas as pd
import feather
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from bayes_opt import BayesianOptimization

# Load Training Data
train = feather.read_dataframe("temp/preparedtraining")

# Load Test Data
test = feather.read_dataframe("temp/preparedtest")

train_labels = np.array(train_raw['loss'])
train_ids = train_raw['id'].values.astype(np.int32)
test_ids = test_raw['id'].values.astype(np.int32)

d_train_full = xgb.DMatrix(x_train, label=train_labels)
d_test = xgb.DMatrix(x_test)

# enter the number of folds from xgb.cv
n_folds = 3
n_rounds = 100

def fitXGBoost(eta = .1, gamma = .5, min_child_weight = 4, colsample_bytree = .3, subsample = 1, max_depth = 6):
    model = xgb.cv({
        "silent": True,
        "learning_rate": eta,
        "gamma": gamma,
        "min_child_weight": min_child_weight,
        "colsample_bytree": colsample_bytree,
        "subsample": subsample,
        "max_depth": int(max_depth),
        "early_stopping_rounds": 20
        }, d_train_full, n_rounds, n_folds)
    return(-model.iloc[-1,0])

bo = BayesianOptimization(fitXGBoost, {
    'eta': (.01, .5),
    'gamma': (0, 4),
    'min_child_weight': (1, 5),
    'colsample_bytree': (.01, 1),
    'subsample': (.5, 1),
    'max_depth': (3, 10)
})

bo.maximize()
