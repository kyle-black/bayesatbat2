import pandas as pd
import sqlite3 as sql
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV

import access_name


# DB Connection
conn = sql.connect('../database/bayesatbat.db')
c = conn.cursor()

# NAME Query TEMP

player_dict = access_name.player_dict

pitch_list = access_name.pitch_list

batter_list = []
for key in player_dict.keys():
    batter_list.append(key)
#name = player_dict.keys()
#index = name.find(',')
# f_name = name[:index] + '\\' + name[index
# print(name)

name = batter_list[80]
#name = name[0]
# print(final_string)
# DB Query


# sql_query = pd.read_sql_query(
#   f'SELECT * from EVENT WHERE batter={name}', conn)

sql_query = pd.read_sql_query(
    f'SELECT * from EVENT', conn)


df = pd.DataFrame(sql_query, columns=['pitch_type', 'p_throws', 'zone', 'release_spin_rate', 'balls', 'strikes',
                                      'release_speed', 'estimated_ba_using_speedangle'])

#'SELECT * from EVENT WHERE player_name=Junis, Jakob'

# print(df)
#########################################

# print(sql_query)
df = df[df['balls'].notna()]
df = df[df['strikes'].notna()]
# print(df.columns)

df = pd.get_dummies(df)
df1 = df

# print(df1)

#df = df[df['launch_speed'].notna()]
df = df[df['estimated_ba_using_speedangle'].notna()]
df = df[df['release_spin_rate'].notna()]

#X = df.drop('player_name')
print(df.columns)
y = df['estimated_ba_using_speedangle'].values


X = df.drop('estimated_ba_using_speedangle', axis=1).values

# print(X)

'''
sc = StandardScaler()

y = y.reshape(-1, 1)


X = sc.fit_transform(X)
y = sc.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=0)

#regressor = RandomForestRegressor(n_estimators=10, random_state=0)
#regressor.fit(X_train, y_train.ravel())

#y_pred = regressor.predict(X_test)

#y_pred = sc.inverse_transform(y_pred, copy=None)


#y_test = sc.inverse_transform(y_test)


GBR = GradientBoostingRegressor(random_state=1)
GBR_parameters = {'learning_rate': [0.01, 0.02, 0.03, 0.04],
                  'subsample': [0.9, 0.5, 0.2, 0.1],
                  'n_estimators': [100, 500, 1000, 1500],
                  'max_depth': [4, 6, 8, 10]
                  }
grid_GBR = GridSearchCV(
    estimator=GBR, param_grid=GBR_parameters, cv=2, n_jobs=-1)


RFR = RandomForestRegressor(random_state=1)
RFR_Grid = {
    "n_estimators": [10, 20, 30],
    "max_features": ["auto", "sqrt", "log2"],
    "min_samples_split": [2, 4, 8],
    "bootstrap": [True, False],
}
grid_GBR = GridSearchCV(estimator=RFR, param_grid=RFR_Grid, cv=2, n_jobs=-1)

LR = LinearRegression()
ereg = VotingRegressor(estimators=[('gb', grid_GBR), ('rf', RFR), ('lr', LR)])
ereg = ereg.fit(X_train, y_train.ravel())

y_pred = ereg.predict(X_test)
y_pred = sc.inverse_transform(y_pred)
y_test = sc.inverse_transform(y_test)
print(y_pred)
#reg1.fit(X, y)
#reg2.fit(X, y)
#reg3.fit(X, y)
print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))

print(df1.columns)
'''
