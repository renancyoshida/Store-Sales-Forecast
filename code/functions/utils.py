import numpy as np

def check_time_split(x, time_split):
    # checks if time split is correct, test dates after train
    for train_index, test_index in time_split.split(x):
        X_train, X_test = x.iloc[train_index, :], x.iloc[test_index,:]

    if max(X_train.index)<=min(X_test.index):
        print('split ok')
    else:
        print('split is not ok')

def rmsle(y, ypred):
    # computes Root Mean Square Log Error
    ypred=np.maximum(ypred, 0)  # avoid log(0)
    
    return np.sqrt(np.mean((np.log1p(y) - np.log1p(ypred))**2))
