import pandas as pd
import matplotlib.pyplot as plt
from pandasql import sqldf
import numpy as np


if __name__ == '__main__':
    data = pd.read_csv("dataset_mood_smartphone.csv")
    ids = sqldf("SELECT DISTINCT id FROM data")
    ids = ids["id"].astype(str).tolist()
    patients = []
    for id in ids:
        patients.append(sqldf("SELECT * FROM data "
               "WHERE id IS '" + id +"'"))
    #print(patients)
    p0 = patients[0]
    p0_mood = sqldf("SELECT variable as mood, value, time FROM p0 "
          "WHERE variable IS 'mood'")
    p0_activity = sqldf("SELECT variable as activity, value, time FROM p0 "
          "WHERE variable IS 'activity'")

    p0_avg_mood = sqldf("SELECT AVG(value) FROM p0_mood "
      "WHERE time BETWEEN '2014-04-19 00:00:00.000' and '2014-04-22 00:00:00.000'")

    print(p0_avg_mood)

