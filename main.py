import pandas as pd
import numpy as np
import requests
from sklearn import preprocessing
import sqlite3 as sql
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc("font", size=14)
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

urls = ["https://www.basketball-reference.com/leagues/NBA_2022_advanced.html",
        "https://www.basketball-reference.com/leagues/NBA_2021_advanced.html",
        "https://www.basketball-reference.com/leagues/NBA_2020_advanced.html",
        "https://www.basketball-reference.com/leagues/NBA_2019_advanced.html",
        "https://www.basketball-reference.com/leagues/NBA_2018_advanced.html",
        "https://www.basketball-reference.com/leagues/NBA_2017_advanced.html",
        "https://www.basketball-reference.com/leagues/NBA_2016_advanced.html",
        "https://www.basketball-reference.com/leagues/NBA_2015_advanced.html",
        "https://www.basketball-reference.com/leagues/NBA_2014_advanced.html",
        "https://www.basketball-reference.com/leagues/NBA_2013_advanced.html",
        "https://www.basketball-reference.com/leagues/NBA_2012_advanced.html"
        ]


if __name__ == '__main__':
    #gamedata = []
    for url in urls:
        x = requests.get(url)
        print(x.content)