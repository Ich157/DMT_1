import pandas as pd
from pandasql import sqldf
import numpy as np
from sklearn.utils import shuffle

data = pd.read_csv("normalised.csv")
data = data.replace({'AS14.': ''}, regex=True)
data["day"] = pd.to_datetime(data["day"])
data["day"] = data["day"].apply(lambda x: x.toordinal())-735313
print(data)
bool_shuffel = False

ids = sqldf("SELECT DISTINCT id FROM data")
ids = ids["id"].astype(str).tolist()

patients = []

for id in ids:
    patients.append(data[data['id'] == id])

list_train = []
list_test = []
for patient in patients:
    if(bool_shuffel):
        patient = shuffle(patient)
    list_train.append(patient[:int(len(patient) *2/3)])
    list_test.append(patient[int(len(patient) *2/3):])

train = pd.concat(list_train)
test = pd.concat(list_test)

train.to_csv("train.csv")
test.to_csv("test.csv")