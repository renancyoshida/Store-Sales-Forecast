import numpy as np
import pandas as pd
import joblib

from constants.paths import data_path, model_path
from prep import data_prep

# load data
df= pd.read_csv(f'{data_path}\\test.csv')
X, y = data_prep(df)

# load model
model_name='xgb_basic'
model = joblib.load(f'{model_path}/{model_name}.pkl')

# make predictions
ypred= np.maximum(model.predict(X),0)

# save predictions
predictions = pd.DataFrame({'id': df['id'], 'sales': ypred})
predictions.to_csv(f'{data_path}/pred-{model_name}.csv', index=False)

