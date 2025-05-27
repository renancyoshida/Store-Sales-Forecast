import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
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
#xgb_scores=-1*cross_val_score(XGBRegressor, X, y, cv=ts, scoring=rmsle_score)


# hyperparameter tuning
import scipy.stats as stats

param_dist = {
    'max_depth': stats.randint(3, 10),
    'learning_rate': stats.uniform(0.01, 0.1),
    'subsample': stats.uniform(0.5, 0.5),
    'n_estimators': stats.randint(50, 200)
}

grid_search = RandomizedSearchCV(XGBRegressor(), param_dist, cv=ts, scoring=rmsle_score)
grid_search.fit(X, y) # performs CV over training data
model=grid_search.best_estimator_

# save model
model_name='xgb_tuned'
joblib.dump(model, f'{model_path}\\{model_name}.pkl')

