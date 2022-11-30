import pandas as pd
import numpy as np
import requests
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc("font", size=14)
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


mvp_stats = pd.read_csv("player_mvp_stats.csv")
mvp_stats = mvp_stats.fillna(0)
del mvp_stats["Unnamed: 0"]
mvp_stats["Player"] = mvp_stats["Player"].str.replace("*", "", regex=False)

#No strings, only numbers
#Cannot use 'Pts Won', 'Pts Max', 'Share' because share is what we are trying to predict
# pts won and pts max are too closly cooralated with share, share = pts won / pts max
predictors = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
       '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year', 'W', 'L', 'W/L%', 'GB', 'PS/G',
       'PA/G', 'SRS']

training = mvp_stats[mvp_stats["Year"] < 2021]
testing = mvp_stats[mvp_stats["Year"] == 2021]

ridge = Ridge(alpha=0.1)

ridge.fit(training[predictors], training["Share"])
predictions = ridge.predict(testing[predictors])

predictions = pd.DataFrame(predictions, columns=["predictions"], index=testing.index)

compare = pd.concat([testing[["Player", "Share"]], predictions], axis=1)
sorted_comp = compare.sort_values("predictions", ascending=False).head(10)
print (sorted_comp)

if __name__ == '__main__':
    pass
