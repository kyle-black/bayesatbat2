import sqlite3 as sql
import pandas as pd

# Databse Connection
conn = sql.connect('../database/bayesatbat.db')
c = conn.cursor()

query = """SELECT DISTINCT batter, player_name from EVENT"""

c.execute(query)
players = c.fetchall()


conn.commit()
# print(player_list[0][0])

player_dict = {}
for name in players:
    player_dict[name[0]] = name[1]

print(player_dict)
