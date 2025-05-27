
import numpy as np
import pandas as pd

from constants.paths import data_path


train=pd.read_csv(f'{data_path}\\test.csv')

stores=pd.read_csv(f'{data_path}\\stores.csv')
oil=pd.read_csv(f'{data_path}\\oil.csv')
holidays_events=pd.read_csv(f'{data_path}\\holidays_events.csv')

df=pd.merge(train, stores, on='store_nbr', how='left')
df=pd.merge(df, oil, on='date', how='left')
df=pd.merge(df, holidays_events[['date', 'transferred']], on='date', how='left')

df['holiday']=1*(df['transferred'] == False) # 1 if not transferred, 0 if transferred
df=df.drop(columns=['transferred'])
df=df.rename(columns={'dcoilwtico': 'oil_price'})

# assuming AR model for oil prices
df['oil_price']=df['oil_price'].ffill() # fill missing oil prices with mean



# == CLEANING AND PREPROCESSING ==
df['date']=pd.to_datetime(df['date'])

df['day_of_week'] = df['date'].dt.dayofweek

# add days since paycheck, since earthquake
df['day'] = df['date'].dt.day
df['days_since_pay']=np.where(df['day']<15, df['day']+1, df['day']-15)
df['days_since_quake']=np.maximum((df['date']-pd.to_datetime('2016-04-16')).dt.days,0)


# turn time frame vars and cluster to object, get dummies
#len(df['family'].unique()) # too many dummies not a concern, if so use embeddings
for var in ['type', 'cluster','day_of_week']:
    df[var] = df[var].astype('object')

cat_cols=[x for x in df.columns if df[x].dtype=='object']

xc=1*pd.get_dummies(df[cat_cols])

# == SPLITS ==
# do cv with time series split and rmsle error using pipeline

if 'sales' in df.columns:
    y=df['sales']
else:
    y='None'

X=pd.concat([df[['onpromotion', 'holiday', 'oil_price','days_since_pay', 'days_since_quake']], xc], axis=1)

