from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd

model = KNeighborsRegressor(n_neighbors=2)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(len(train))
print(len(test))
train_X = train[["activity","screen","valence","arousal","office","game","call","sms","social","entertainment","communication","weather"]]
train_Y = train[["mood"]]
test_X = test[["activity","screen","valence","arousal","office","game","call","sms","social","entertainment","communication","weather"]]
test_Y = test[["mood"]].to_numpy()
model.fit(train_X.to_numpy(),train_Y.to_numpy())
predictions = model.predict(test_X.to_numpy())
correct = 0
for i in range(len(predictions)):
    if test_Y[i] == predictions[i]:
        correct = correct + 1
    else:
        print(test_Y[i]-predictions[i])
print(correct)


