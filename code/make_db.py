import sqlite3 as sql
import pandas as pd

# Databse Connection
conn = sql.connect('../database/bayesatbat.db')
c = conn.cursor()


# Pandas Connection
df = pd.read_csv('../database/first_half.csv')
#df2 = pd.read_csv('../database/2021_Ball_in_Play_2.csv')

#df = df.merge(df2)

df.to_sql('EVENT', con=conn, if_exists='replace')
print(df)


conn.commit()
