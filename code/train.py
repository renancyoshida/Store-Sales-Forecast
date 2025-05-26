import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor

from functions.utils import check_time_split, rmsle
from constants.paths import  data_path, model_path
from prep import data_prep

# load data
train=pd.read_csv(f'{data_path}\\train.csv')
X,y= data_prep(train)


# == MODELING ==
# try linear regression, random forest, xgboost, lightgbm, catboost

ts=TimeSeriesSplit(n_splits=3)
rmsle_score=make_scorer(rmsle, greater_is_better=False)
xgb_scores=-1*cross_val_score(XGBRegressor(), X, y, cv=ts, scoring=rmsle_score)

print('XGB scores:', xgb_scores)

model=XGBRegressor()
model.fit(X,y) # is this correct?

# save model
joblib.dump(model, f'{model_path}/xgb_basic.pkl')

