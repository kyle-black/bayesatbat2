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


import access_name


# DB Connection
conn = sql.connect('../database/bayesatbat.db')
c = conn.cursor()

# NAME Query TEMP

player_dict = access_name.player_dict

batter_list = []
for key in player_dict.keys():
    batter_list.append(key)
#name = player_dict.keys()
#index = name.find(',')
# f_name = name[:index] + '\\' + name[index
# print(name)

name = batter_list[10]
#name = name[0]
# print(final_string)
# DB Query

sql_query = pd.read_sql_query(
    f'SELECT * from EVENT WHERE batter={name}', conn)


df = pd.DataFrame(sql_query, columns=['player_name', 'pitch_type', 'p_throws', 'launch_speed',
                                      'launch_angle', 'release_speed', 'description', 'estimated_ba_using_speedangle'])

#'SELECT * from EVENT WHERE player_name=Junis, Jakob'

print(df)
#########################################

# print(sql_query)


df = df[df['launch_speed'].notna()]
df = df[df['estimated_ba_using_speedangle'].notna()]


y = df['estimated_ba_using_speedangle'].values


X = df.drop('estimated_ba_using_speedangle', axis=1).values


sc = StandardScaler()

y = y.reshape(-1, 1)


X = sc.fit_transform(X)
y = sc.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=0)

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

y_pred = sc.inverse_transform(y_pred, copy=None)


y_test = sc.inverse_transform(y_test)


reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()
ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
ereg = ereg.fit(X_train, y_train)

y_pred = ereg.predict(X_test)


reg1.fit(X, y)
reg2.fit(X, y)
reg3.fit(X, y)
