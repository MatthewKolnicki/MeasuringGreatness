import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor


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
training = mvp_stats[mvp_stats["Year"] < 2011]
testing = mvp_stats[mvp_stats["Year"] == 2011]




def regression(model):
    model.fit(training[predictors], training["Share"])
    predictions = model.predict(testing[predictors])

    predictions = pd.DataFrame(predictions, columns=["predictions"], index=testing.index)
    return predictions


def top5_compared(predictions):
    compare = pd.concat([testing[["Player", "Share"]], predictions], axis=1)
    sorted_comp = compare.sort_values("predictions", ascending=False).head(10)
    return sorted_comp


def accuracy(sorted_comp):
    actual = sorted_comp.sort_values("Share", ascending=False).head(5)
    predictions = sorted_comp.sort_values("predictions", ascending=False)
    checker = []
    found = 0
    seen = 1
    for index, row in predictions.iterrows():
        if row["Player"] in actual["Player"].values:
            found += 1
            checker.append(found / seen)
        seen += 1

    top5 = sum(checker) / len(checker)
    return top5


def accuracy_mvp(sorted_comp):
    actual = sorted_comp.sort_values("Share", ascending=False).head(1)
    predictions = sorted_comp.sort_values("predictions", ascending=False).head(1)

    if actual.iloc[0]['Player'] == predictions.iloc[0]['Player']:
        return 1
    return 0


def compound_years_test(model, year, predictions):
    average_accuracy = []
    compound_prediction = []
    average_accuracy_mvp = []
    year_by_year_mvp = []
    year_by_year = []
    years_list = []
    for year in years[5:]:
        training = mvp_stats[mvp_stats["Year"] < year]
        testing = mvp_stats[mvp_stats["Year"] == year]
        model.fit(training[predictors], training["Share"])
        predictions = model.predict(testing[predictors])
        predictions = pd.DataFrame(predictions, columns=["predictions"], index=testing.index)
        compare = pd.concat([testing[["Player", "Share"]], predictions], axis=1)
        compound_prediction.append(compare)
        years_list.append(year)
        average_accuracy.append(accuracy(compare))
        year_by_year.append(accuracy(compare))
        average_accuracy_mvp.append(accuracy_mvp(compare))
        year_by_year_mvp.append(accuracy_mvp(compare))
        mvp_compound = (sum(average_accuracy_mvp) / len(average_accuracy_mvp)) * 100
        year_by_year = [float(('%.4f' % float(elem)).strip("0")) for elem in year_by_year]
        top5_compound = (sum(average_accuracy) / len(average_accuracy)) * 100
    print("Years in order =", years_list)
    print("MVP hit or miss by year =", year_by_year_mvp)
    print("MVP predicted accuracy per year", '%.4f' % mvp_compound, "%")
    print("Top 5 accuracy by year =", year_by_year)
    print("Accuracy of top 5 voted players compounded", '%.2f' % top5_compound, "%")
    #print(pd.concat([pd.Series(model.coef_), pd.Series(predictors)], axis=1).sort_values(0, ascending=False))



if __name__ == '__main__':
    ridge = Ridge(alpha=.1)
    las = Lasso(alpha=.1)
    lin = LinearRegression()
    elas = ElasticNet(alpha=.1)
    sgd = SGDRegressor(alpha=.1)
    rf = RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5)

    years = list(range(1991, 2022))
    # First 5 years are training first test set (1996), next is 6, then 7, ect.

    print("Ridge")
    #compound_years_test(ridge, years[5:], regression(ridge))
    print("\nLasso")
    #compound_years_test(las, years[5:], regression(las))
    print("\nLinear")
    #compound_years_test(lin, years[5:], regression(lin))
    print("\nElastic Net")
    #compound_years_test(elas, years[5:], regression(elas))
    print("\nSGD")
    #compound_years_test(sgd, years[5:], regression(sgd))
    print("\nRandom Forest")
    compound_years_test(rf, years[5:], regression(rf))