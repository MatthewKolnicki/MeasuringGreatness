import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# mvps
mvps = pd.read_csv("mvps.csv")
mvps = mvps[["Player", "Year", "Pts Won", "Pts Max", "Share"]]

# players
players = pd.read_csv("players.csv")
# Makes sure cols are filled
del players["Unnamed: 0"]
# Dels Rank col why??
del players["Rk"]
# Replaces * with empty in player col
players["Player"] = players["Player"].str.replace("*", "", regex=False)


# SpecificPlayer = players[players["Player"] == "Greg Anderson"]

# Prints players seperated by teams
def single_team(df):
    if df.shape[0] == 1:
        return df
    else:
        row = df[df["Tm"] == "TOT"]
        row["Tm"] = df.iloc[-1, :]["Tm"]
        return row


# players = players.groupby(["Player", "Year"]).apply(single_team)

# Combines mvps and players dbs that match player and year cols
combined = players.merge(mvps, how="outer", on=["Player", "Year"])

# outputs combined db and only rows who have pts > num
pts_check = combined[combined["Pts Won"] > 25]

mvp_stats = pd.read_csv("player_mvp_stats.csv")

print (mvp_stats)

# players.index = players.index.droplevel() #Will drop first row in db


if __name__ == '__main__':
    pass
