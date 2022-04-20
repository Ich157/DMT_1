from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
from sklearn import linear_model

model = KNeighborsRegressor(n_neighbors=3)
train = pd.read_csv("train_shuffeld.csv")
test = pd.read_csv("test_shuffeld.csv")
train_X = train[["activity","screen","valence","arousal","office","game","call","sms","social","entertainment","communication","weather"]]
train_Y = train[["mood"]]
test_X = test[["activity","screen","valence","arousal","office","game","call","sms","social","entertainment","communication","weather"]]
test_Y = test[["mood"]].to_numpy()
print(np.all(np.isfinite(train_X)))
print(np.any(np.isnan(train_X)))
model.fit(train_X.to_numpy(),train_Y.to_numpy())
predictions = model.predict(test_X.to_numpy())
correct = 0
error = []
for i in range(len(predictions)):
    if test_Y[i] == predictions[i]:
        correct = correct + 1
    error.append(test_Y[i]-predictions[i])
print(error)
print(correct)
print(len(predictions))


