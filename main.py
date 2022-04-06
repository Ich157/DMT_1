import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandasql import sqldf
import numpy as np
if __name__ == '__main__':
    data = pd.read_csv("dataset_mood_smartphone.csv") # 376912 rows x 5 columns
    data = data.drop(data.columns[[0]], axis = 1)

    print(data)

    newdf = sqldf("SELECT id, time FROM data") # 376912 rows x 2 columns
    #print(newdf)


    # # HOW DO WE GET IT ALL IN COLUMNS
    for attr in set(data["variable"].values):
         fill = data["variable"].values
         #mask = data.mask(["variable"] == attr)
         new_col = np.empty(shape = len(fill))
         new_col[:] = np.nan
         for i, row in enumerate(fill):

             if row == attr:
                 new_col[i] = data.iloc[i]["value"]

         newdf[attr] = new_col

    print(newdf)
    newdf.to_csv('newdf.csv')
    #

