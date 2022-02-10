import pandas as pd

player_df = pd.read_csv('MAIN.csv')
print(player_df[['batter', 'player_name']].unique())
