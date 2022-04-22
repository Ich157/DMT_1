import pandas as pd
import numpy as np
def previous_day(moods):
    abse = []
    predictions = []
    for i, _ in enumerate(moods):
        if i < len(moods) -1:
            abse.append(abs(moods[i] - moods[i+1]))
            predictions.append(moods[i])

    return np.mean(abse), predictions, abse


if __name__ == "__main__":
    test = pd.read_csv("test.csv")

    moods = test["mood"].values
    print(moods)
    acc, preds, abse = previous_day(moods)
    print(acc)
    print(preds)
    print(abse)
    print(len(preds))
