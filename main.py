import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandasql import sqldf
import numpy as np

def get_done_df():
    data = pd.read_csv("dataset_mood_smartphone.csv")  # 376912 rows x 5 columns
    data = data.drop(data.columns[[0]], axis=1)
    newdf = sqldf("SELECT id, time FROM data")  # 376912 rows x 2 columns
    # print(newdf)

    # # HOW DO WE GET IT ALL IN COLUMNS
    for attr in set(data["variable"].values):
        fill = data["variable"].values
        # mask = data.mask(["variable"] == attr)
        new_col = np.empty(shape=len(fill))
        new_col[:] = np.nan
        for i, row in enumerate(fill):

            if row == attr:
                new_col[i] = data.iloc[i]["value"]

        newdf[attr] = new_col
    print(newdf.columns)
    newdf.rename(columns={'appCat.weather': 'weather', 'appCat.social': 'social', 'appCat.travel': 'travel',
                          'appCat.finance': 'finance',
                          'circumplex.valence': 'valence', 'appCat.utilities': 'utilities', 'appCat.other': 'other',
                          'appCat.communication': 'communication',
                          'appCat.game': 'game', 'appCat.builtin': 'builtin', 'appCat.office': 'office',
                          'appCat.entertainment': 'entertainment',
                          'circumplex.arousal': 'arousal', 'appCat.unknown': 'unknown'}, inplace=True)
    print(newdf.columns)
    newdf['day'] = pd.to_datetime(newdf['time']).dt.date

    print(sqldf("SELECT AVG(mood) as mood FROM newdf GROUP BY id, day"))
    print(max(newdf['mood']))

    donedf = sqldf("SELECT id, day, SUM(screen) as screen, SUM(call) as call, SUM(social) as social, SUM(sms) as sms, "
                   "SUM(builtin) as builtin, SUM(utilities) as utilities, AVG(arousal) as arousal, "
                   "SUM(finance) as finance, SUM(unknown) as unknown, AVG(valence) as valence, "
                   "SUM(office) as office, AVG(activity) as activity, SUM(game) as game, SUM(entertainment) as entertainment, "
                   "SUM(weather) as weather, SUM(communication) as communication, SUM(travel) as travel, "
                   "SUM(other) as other, AVG(mood) as mood FROM newdf "
                   "GROUP BY id, day")

    donedf.to_csv('donedf.csv')


def skip_start(start = "2014-03-20"):

    donedf = pd.read_csv("donedf.csv")
    donedf = donedf.drop(donedf.columns[[0]], axis=1)
    skip_start = donedf[(donedf['day'] > start)]
    print(skip_start)
    skip_start.to_csv("skip_start.csv")


def dropnans(df, columns):
    for col in columns:
        df[col] = df[col].fillna(0)

    return df

def shift_mood(df):
    ids = sqldf("SELECT DISTINCT id FROM df")
    ids = ids["id"].astype(str).tolist()

    patients = []

    for id in ids:
        patients.append(df[df['id'] == id])

    list_shifted = []
    for patient in patients:
        patient['mood'] = patient['mood'].shift(-1)
        patient = patient[:-1]
        list_shifted.append(patient)
    return pd.concat(list_shifted)

def drop_mood_nan(df):
    return df[df['mood'].notna()]

def fill_nan(df):
    mean_activity = df["activity"].mean()
    df["activity"].fillna(mean_activity)
    mean_arousal = df["arousal"].mean()
    df["arousal"].fillna(mean_arousal)
    mean_valence = df["valence"].mean()
    df["valence"].fillna(mean_valence)
    return df.fillna(0)


if __name__ == '__main__':
    skip_start = pd.read_csv("skip_start.csv")
    to_drop = ["call", "sms", "utilities", "game", "unknown", "finance", "office", "weather", "travel"]
    dropped_nan = dropnans(skip_start, to_drop)
    print(dropped_nan.isnull().sum())
    print(dropped_nan.shape)
    dropped_nan.to_csv("dropped_nan.csv")
    shifted = shift_mood(dropped_nan)
    shifted.to_csv("shifted.csv")
    dropped_mood_nan = drop_mood_nan(shifted)
    dropped_mood_nan.to_csv("dropped_mood_nan.csv")
    filled_nan = fill_nan(dropped_mood_nan)
    filled_nan.to_csv("filled_nan.csv")