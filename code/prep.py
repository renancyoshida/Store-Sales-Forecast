
import numpy as np
import pandas as pd

from constants.paths import data_path


# TO DO: add oil, holidays

def data_prep(data):
    stores=pd.read_csv(f'{data_path}\\stores.csv')


    df=pd.merge(data, stores, on='store_nbr', how='left')

    # == CLEANING AND PREPROCESSING ==

    # check for missing values
    #missing = df.isnull().sum()
    #missing = missing[missing > 0]

    # get month, day of week, year
    #df['month'] = pd.to_datetime(df['date']).dt.month
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    #df['year'] = pd.to_datetime(df['date']).dt.year

    df.drop(columns=['date'], inplace=True)

    # turn time frame vars and cluster to object, get dummies

    #len(df['family'].unique()) # too many dummies not a concern, if so use embeddings

    for var in ['type', 'cluster','day_of_week']:
        df[var] = df[var].astype('object')

    cat_cols=[x for x in df.columns if df[x].dtype=='object']

    xc=1*pd.get_dummies(df[cat_cols])

    # for j in xc.columns:
    #     print(j)

    # == SPLITS ==
    # do cv with time series split and rmsle error using pipeline
    
    if 'sales' in df.columns:
        y=df['sales']
    else:
        y='None'

    X=pd.concat([df['onpromotion'], xc], axis=1)

    return X, y

if __name__ == '__main__':
    # load data
    train=pd.read_csv(f'{data_path}\\train.csv')
    test=pd.read_csv(f'{data_path}\\test.csv')
    
    # prep data
    X_train, y_train = data_prep(train)
    X_test = data_prep(test) # what if categorical vars not in train?


