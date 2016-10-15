#!/usr/bin/env python3

import numpy as np
import pandas as pd
import feather
import xgboost as xgb
from bayes_opt import BayesianOptimization

# Load Training Data
train = feather.read_dataframe("temp/preparedtraining.feather")
train_labels = np.array(train['loss'])
train.drop(train.columns[[-1,-2]], 1, inplace = True)
d_train_full = xgb.DMatrix(train, label=train_labels)

# if you're paranoid about overfitting, increase this.
n_folds = 3

# if you see metrics dropping precipitously until the end, increase this.
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
        }, d_train_full, n_rounds, n_folds, metrics = ["mae"])
    return(-model.iloc[-1,0])

bo = BayesianOptimization(fitXGBoost, {
    'eta': (.01, .5),
    'gamma': (0, 4),
    'min_child_weight': (1, 5),
    'colsample_bytree': (.01, 1),
    'subsample': (.5, 1),
    'max_depth': (3, 12)
})

bo.maximize(init_points = 60, n_iter = 120)
