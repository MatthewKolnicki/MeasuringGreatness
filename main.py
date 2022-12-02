import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

plt.style.use('seaborn')

mvp_stats = pd.read_csv("player_mvp_stats.csv")
mvp_stats = mvp_stats.fillna(0)
del mvp_stats["Unnamed: 0"]
mvp_stats["Player"] = mvp_stats["Player"].str.replace("*", "", regex=False)

figure_num = 1

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
    for year in year:
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
        year_by_year = [float(('%.3f' % float(elem)).strip("0")) for elem in year_by_year]
        top5_compound = (sum(average_accuracy) / len(average_accuracy)) * 100
        str_model = str(model)
        sep = str_model.split("(", 1)
        str_model = sep[0]
        print("Running " + str_model + " for year " + str(year) + ", Top 5 Accuracy = "
              + '%.3f' % (accuracy(compare) * 100) + "%"
              + ", MVP Accuracy = " + str(accuracy_mvp(compare) * 100) + "%")
    # print("Years in order =", years_list)
    # print("MVP hit or miss by year =", year_by_year_mvp)
    # print("MVP predicted accuracy per year", '%.4f' % mvp_compound, "%")
    # print("Top 5 accuracy by year =", year_by_year)

    str_model = str(model)
    sep = str_model.split("(", 1)
    str_model = sep[0]
    global figure_num
    plt.figure(figure_num)
    plt.scatter(years[5:], year_by_year, c='blue', edgecolors='black', linewidths=1, alpha=0.7)
    # plt.xticks(years[5:])
    plt.xlabel("Years")
    plt.ylabel("Accuracy of Top 5 MVP Prediction")
    plt.title(str_model + ": Top 5 MVP Prediction")
    plt.savefig(("Results/" + str_model + "_TOP5.png"))

    figure_num = figure_num + 1

    plt.figure(figure_num)
    plt.scatter(years[5:], year_by_year_mvp, c='red', edgecolors='black', linewidths=1, alpha=0.7)
    # plt.xticks(years[5:])
    plt.xlabel("Years")
    plt.ylabel("Accuracy MVP Prediction")
    plt.title(str_model + ": MVP Prediction")
    plt.savefig(("Results/" + str_model + "_MVP.png"))

    figure_num = figure_num + 1

    return '%.3f' % top5_compound + "%        |    " + '%.3f' % mvp_compound + "%"

    # return year_by_year
    # print(pd.concat([pd.Series(model.coef_), pd.Series(predictors)], axis=1).sort_values(0, ascending=False))


if __name__ == '__main__':
    #Definition for Linear Models
    ridge = Ridge(alpha=.1)
    las = Lasso(alpha=.001)
    lin = LinearRegression()
    elas = ElasticNet(alpha=.1)
    sgd = SGDRegressor()
    #Definition for
    rf = RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5)

    years = list(range(1991, 2022))
    # First 5 years are training first test set (1996), next is 6, then 7, ect.

    data = {'Model': ['Ridge', 'Lasso', 'Linear', 'Elastic Net', 'SGD', 'Random Forest'],
            'Top 5 Accuracy | MVP Accuracy': [y := compound_years_test(ridge, years[5:], regression(ridge)),
                                              h := compound_years_test(las, years[5:], regression(las)),
                                              compound_years_test(lin, years[5:], regression(lin)),
                                              compound_years_test(elas, years[5:], regression(elas)),
                                              compound_years_test(sgd, years[5:], regression(sgd)),
                                              compound_years_test(rf, years[5:], regression(rf))],
            }


    df = pd.DataFrame(data)
    print((tabulate(df, headers='keys', tablefmt='psql')))
    plt.show()