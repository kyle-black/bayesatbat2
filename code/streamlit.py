import pandas as pd
import sqlite3 as sql
import numpy as np
#import matplotlib.pyplot as plt
#from scipy.fftpack import cs_diff, sc_diff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import streamlit as st

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV

import matplotlib.image as mpimg

#import access_name
import plotly.express as px
#import code.streamlit_app as st
#import access_name
import requests
import urllib
#import cv2
import os

import subprocess
import sys


#csv_path = os.path.abspath("MAIN.csv")
#import pandasql
csv_path = ('MAIN.csv')
#subprocess.run([f"{sys.executable}", "access_name.py"])

#import time

#my_bar = st.progress(0)
# st.write(csv_path)

##############REMOVE PREVIOUS PLAYER IMAGES ###############
dir = 'images/players/'
try:
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
except:
    print('no picture')
################################################

html_string1 = '''<body style="background-color:powderblue;">

<h1>This is a heading</h1>
<p>This is a paragraph.</p>

</body>'''


html_string = '<h1 style="font-family: Garamond, serif;">Simulated Batting Average After Contact</h1>'
st.markdown(html_string, unsafe_allow_html=True)

#pitch_list = access_name.pitch_list
# DB Connection
#db_connect = 'bayesatbat.db'
#conn = sql.connect(db_connect)
#c = conn.cursor()
# st.write(conn)

######################################################

player_df = pd.read_csv(csv_path)
# print(player_list[0][0])


player_df = player_df[['batter', 'player_name']]
player_df = player_df.drop_duplicates()


player_dict = {}
# for name in players:
#   player_dict[name[0]] = name[1]

# print(player_dict)

for name, row in player_df[:].iterrows():

    player_dict[row['batter']] = row['player_name']
    #print(row['batter'], row['player_name'])


#####################################################
# NAME Query TEMP

#player_dict = access_name.player_dict

batter_list = []
for key in player_dict.keys():
    batter_list.append(key)

# Streamlit Column
#st.header('Expected Batting Average When Making Contact')
with st.expander("Model explanation"):
    st.write("""
         Data derived from SQLite3 database using 2021 players with plate apperarces > 200.
         This model uses Gradient Boosted Regression, Random Forest Regression, and Linear Regression
         and takes the mean of all three predictions using optimized parameters (Grid Search).
         Using the simulated pitching parameters model predicts the batting average AFTER contact.
     """)


col1, col2, col3 = st.columns(3)


################################Streamlit Select List Side Bar ##################
################################
#batter_tuple = tuple(player_dict.value(), player_dict.keys())
st.sidebar.image('images/MLB.png')

option = st.sidebar.selectbox('Select Batter...', player_dict.values())

#st.write('Player selected: ', option)
#########################################Retreive PLAYER IMAGES ####################
player_df = pd.read_excel('database/Player-ID-Map.xlsx')
player_df = player_df[['NFBCLASTFIRST', 'ESPNID']]

player_df = player_df[player_df['NFBCLASTFIRST'] == option]

#player = int(player_df['ESPNID'])

player_df["player"] = pd.to_numeric(player_df["ESPNID"])

player = int(player_df["player"])

# st.write(player)
filename = f'images/players/{player}.png'
image_url = f"https://a.espncdn.com/combiner/i?img=/i/headshots/mlb/players/full/{player}.png"

# calling urlretrieve function to get resource
urllib.request.urlretrieve(image_url, filename)
img = mpimg.imread(filename)
# st.write()


# with col2:


####################################################################################
player_serial = list(player_dict.keys())[
    list(player_dict.values()).index(option)]
#st.write('serial:', result)
#################################


st.sidebar.title('Simulated Pitch Options')


hand = st.sidebar.radio('Pitcher Handedness:', ('Right', 'Left'))
if hand == 'Right':
    st.sidebar.write('Right Handed')
    throws = 'R'
    right = 1
else:
    right = 0
if hand == 'Left':
    st.sidebar.write('Left Handed')
    throws = 'L'
    left = 1
else:
    left = 0

pitch_type = st.sidebar.selectbox('Select Pitch Type...', ('Four-Seam Fastball', 'Slider', 'Change-Up', 'Cut Fastball',
                                                           'Curveball', 'Eephus', 'Forkball', 'Splitter', 'Two-Seam Fastball', 'Knuckle Curve', 'Knuckle Ball', 'Sinker', 'Fastball(general)'))

if pitch_type == 'Four-Seam Fastball':
    pitch = 'FF'
    ff = 1
else:
    ff = 0
if pitch_type == 'Slider':
    pitch = 'SL'
    sl = 1
else:
    sl = 0
if pitch_type == 'Change-Up':
    pitch = 'CH'
    ch = 1
else:
    ch = 0

if pitch_type == 'Cut Fastball':
    pitch = 'FC'
    fc = 1
else:
    fc = 0
if pitch_type == 'Curveball':
    pitch = 'CU'
    cu = 1
else:
    cu = 0
if pitch_type == 'Eephus':
    pitch = 'EP'
    ep = 1
else:
    ep = 0
if pitch_type == 'Forkball':
    pitch = 'FO'
    fo = 1
else:
    fo = 0
if pitch_type == 'Splitter':
    pitch = 'FS'
    fs = 1
else:
    fs = 0
if pitch_type == 'Two-Seem Fastball':
    pitch = 'FT'
    ft = 1
else:
    ft = 0
if pitch_type == 'Knuckle Curve':
    pitch = 'KC'
    kc = 1
else:
    kc = 0

if pitch_type == 'Knuckle':
    pitch = 'KN'
    kn = 1
else:
    kn = 0
if pitch_type == 'Sinker':
    pitch = 'SI'
    si = 1
else:
    si = 0
if pitch_type == 'Fastball(general)':
    pitch = 'FA'
    fa = 1
else:
    fa = 0

if pitch_type == 'Slow-Curveball':
    pitch = 'CS'
    cs = 1
else:
    cs = 0

if pitch_type == 'Screwball':
    pitch = 'sc'
    sc = 1
else:
    sc = 0

########################################

st.sidebar.write('Pitch:', pitch_type)
##########################################


###########################################


zone = st.sidebar.selectbox('Select Pitch Zone', ('1', '2', '3', '4',
                                                  '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'))


#####################################

speed = st.sidebar.slider('Pitch Speed (MPH):', 50, 105, 90)
st.sidebar.write('Pitch Speed (MPH):', speed)


spin = st.sidebar.slider('Release Spin Rate (RPM):', 100, 3600, 2500)

st.sidebar.write('Spin Rate (RPM):', spin)

st.sidebar.title('Count:')

ball = st.sidebar.slider('Ball:', 0, 3, 1)

st.sidebar.write('Balls:', ball)

strike = st.sidebar.slider('Strike:', 0, 2, 1)

st.sidebar.write('Strikes:', strike)

simulated_pitch = [pitch, throws, zone, spin, ball, strike, speed]


with col1:
    st.image(img)
    st.subheader(f'  {option}')

with col2:
    st.write(f'Pitcher Hand: {throws}')
    st.write(f'Pitch Type: {pitch}')
    st.write(f'Pitch Speed: {speed} MPH')
    st.write(f'Pitch Spin: {spin} RPM')
    st.write(f'Balls: {ball}')
    st.write(f'Strikes: {strike}')
    if zone == '1':
        st.image('/images/strikezone/1.png')

    if zone == '2':
        st.image('/images/strikezone/2.png')

    if zone == '3':
        st.image('/images/strikezone/3.png')

    if zone == '4':
        st.image('/images/strikezone/4.png')

    if zone == '5':
        st.image('/images/strikezone/5.png')

    if zone == '6':
        st.image('/images/strikezone/6.png')

    if zone == '7':
        st.image('/images/strikezone/7.png')

    if zone == '8':
        st.image('/images/strikezone/8.png')

    if zone == '9':
        st.image('/images/strikezone/9.png')

    if zone == '10':
        st.image('/images/strikezone/10.png')

    if zone == '11':
        st.image('/images/strikezone/11.png')

    if zone == '12':
        st.image('/images/strikezone/12.png')

    if zone == '13':
        st.image('/images/strikezone/13.png')

    if zone == '14':
        st.image('/images/strikezone/14.png')


###################################
hold_array = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

hold_array[0] = zone
hold_array[1] = spin
hold_array[2] = ball
hold_array[3] = strike
hold_array[4] = speed
hold_array[5] = ch
hold_array[6] = cs
hold_array[7] = cu
hold_array[8] = fa
hold_array[9] = fc
hold_array[10] = ff
hold_array[11] = fs
hold_array[12] = kc
hold_array[13] = kn
hold_array[14] = sc
hold_array[15] = si
hold_array[16] = sl
hold_array[17] = left
hold_array[18] = right


###################################

# DB Query

# sql_query = pd.read_sql_query(
#   f'SELECT * from EVENT', conn)


sim_dict = {'pitch_type': pitch, 'p_throws': throws, 'zone': zone,
            'release_spin_rate': spin, 'balls': ball, 'strikes': strike, 'release_speed': speed}


####################################################
###################################################
# df = pd.DataFrame(sql_query, columns=['batter', 'pitch_type', 'p_throws', 'zone', 'release_spin_rate', 'balls', 'strikes',
#                                     'release_speed', 'estimated_ba_using_speedangle'])

# df = pd.DataFrame(sql_query, columns=['batter', 'pitch_type', 'p_throws', 'zone', 'release_spin_rate', 'balls', 'strikes',
#                                     'release_speed', 'estimated_ba_using_speedangle'])

df = pd.read_csv(csv_path)

df = df[['batter', 'pitch_type', 'p_throws', 'zone', 'release_spin_rate', 'balls', 'strikes',
         'release_speed', 'estimated_ba_using_speedangle']]

#'SELECT * from EVENT WHERE player_name=Junis, Jakob'
df = pd.get_dummies(df)
y = df['estimated_ba_using_speedangle']
df = df[df['batter'] == player_serial]
df = df.drop('batter', axis=1)

# st.write(sim_dict)


# 3st.dataframe(df)
#########################################
#########################################
# print(sql_query)
df = df[df['balls'].notna()]
df = df[df['strikes'].notna()]

#df = df.append(simulated_pitch)
#df = pd.get_dummies(df)

#df = df[df['launch_speed'].notna()]
df = df[df['estimated_ba_using_speedangle'].notna()]
df = df[df['release_spin_rate'].notna()]
#sim_pitch = df.iloc[-1]
#df = df[:-1]
#X = df.drop('player_name')
# st.write(df)
y = df['estimated_ba_using_speedangle'].values

# st.write(df)
X = df.drop('estimated_ba_using_speedangle', axis=1).values


y = y.reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=0)


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
#y_pred = sc.inverse_transform(y_pred)
#y_test = sc.inverse_transform(y_test)
# print(y_pred)


#reg1.fit(X, y)
#reg2.fit(X, y)
#reg3.fit(X, y)
print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))

#########################################################
#########################################################
# st.write(y_pred)
# st.write(y_test)
# st.write(X_test)
#st.write(metrics.mean_squared_error(y_test, y_pred))


#sim_pitch = sim_pitch.values
sim_pitch = hold_array

sim_pitch = sim_pitch.reshape(1, -1)
#sim_pitch = sc.fit_transform(test)

pred = ereg.predict(sim_pitch)
if pred[0] < 0:
    pred = '.000'

else:
    pred = round(pred[0], 3)


#pred = sc.inverse_transform(pred)
with col3:
    st.subheader('Simulated Batting Average:')

    html_string2 = f'<h1 style="font-family: Garamond, serif;color:green;">  {pred}</h1>'
    st.markdown(html_string2, unsafe_allow_html=True)

    RMSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = round(RMSE, 3)
    st.subheader('Error(RMSE):')
    html_string3 = f'<h1 style="font-family: Garamond, serif;color:red;">  {RMSE}</h1>'
    st.markdown(html_string3, unsafe_allow_html=True)

    # st.subheader(f'{pred}')


#prediction = sc.inverse_transform(pred)
# print(sim_pitch.shape)
# st.write(X_test[0])
###########################Streamlit Footer header ########

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
