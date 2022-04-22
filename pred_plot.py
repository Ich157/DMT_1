import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    data = pd.read_csv('error_LSTM.csv')
    train_data = pd.read_csv('train.csv')
    print(data.head())
    print(train_data.columns)
    print(data.columns)
    print(data["day"])
    #data["prediction"] = data["prediction"].numpy()
    #data["id"] = data["id"].numpy()
    #(data.groupby("id")["prediction"].plot())


    plt.figure()
    data.groupby("id")["prediction"].plot()
    data.groupby("id")["target"].plot()
    plt.show()

    for i in data["id"].unique():
        print(i)
        patientx = data[data["id"] == i]
        patientprev = train_data[train_data["id"] == i]
        #print(patientx.head())
        #patientprev.to_csv("wow.csv")
        #plt.figure()
        #print(type(patientprev["mood"]))
        #print(patientprev["mood"].shape)

        x= np.arange(1,44)
        #print(x)
        #print(patientprev["mood"])
        fig, ax = plt.subplots()
        ax.plot(patientprev["day"], patientprev["mood"], 'b')
        ax.plot(patientx["day"], patientx["prediction"], 'g')
        ax.plot(patientx["day"], patientx["target"], 'y')
        ax.legend(["Training mood", "Predicted mood", "Actual mood"])
        plt.xlabel("day")
        plt.title("Patient " + str(i) + " mood over the days")
        plt.show()
