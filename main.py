import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
plt.rc("font", size=14)

mvp_stats = pd.read_csv("player_mvp_stats.csv")
mvp_stats = mvp_stats.fillna(0)
del mvp_stats["Unnamed: 0"]
mvp_stats["Player"] = mvp_stats["Player"].str.replace("*", "", regex=False)

# No strings, only numbers
# Cannot use 'Pts Won', 'Pts Max', 'Share' because share is what we are trying to predict
# pts won and pts max are too closly cooralated with share, share = pts won / pts max
predictors = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
              '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
              'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year', 'W', 'L', 'W/L%', 'GB', 'PS/G',
              'PA/G', 'SRS']

training = mvp_stats[mvp_stats["Year"] != 2021]
testing = mvp_stats[mvp_stats["Year"] == 2021]


def ridge_regression():
    ridge = Ridge(alpha=.1)

    ridge.fit(training[predictors], training["Share"])
    predictions = ridge.predict(testing[predictors])

    predictions = pd.DataFrame(predictions, columns=["predictions"], index=testing.index)

    compare = pd.concat([testing[["Player", "Share"]], predictions], axis=1)
    sorted_comp = compare.sort_values("predictions", ascending=False).head(10)
    print("Ridge", sorted_comp, '\n')

def lasso_regression():
    las = Lasso(alpha=.001)

    las.fit(training[predictors], training["Share"])
    predictions = las.predict(testing[predictors])

    predictions = pd.DataFrame(predictions, columns=["predictions"], index=testing.index)

    compare = pd.concat([testing[["Player", "Share"]], predictions], axis=1)
    sorted_comp = compare.sort_values("predictions", ascending=False).head(10)
    print("Lasso", sorted_comp, '\n')

def linear_regression():
    lin = LinearRegression()

    lin.fit(training[predictors], training["Share"])
    predictions = lin.predict(testing[predictors])

    predictions = pd.DataFrame(predictions, columns=["predictions"], index=testing.index)

    compare = pd.concat([testing[["Player", "Share"]], predictions], axis=1)
    sorted_comp = compare.sort_values("predictions", ascending=False).head(10)
    print("Linear", sorted_comp, '\n')

def elastic_regression():
    elas = ElasticNet(alpha=.1)

    elas.fit(training[predictors], training["Share"])
    predictions = elas.predict(testing[predictors])

    predictions = pd.DataFrame(predictions, columns=["predictions"], index=testing.index)

    compare = pd.concat([testing[["Player", "Share"]], predictions], axis=1)
    sorted_comp = compare.sort_values("predictions", ascending=False).head(10)
    print("Elastic", sorted_comp, '\n')

if __name__ == '__main__':
    ridge_regression()
    lasso_regression()
    linear_regression()
    elastic_regression()