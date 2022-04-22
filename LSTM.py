import time

import matplotlib.pyplot
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from copy import deepcopy
import numpy as np
import pandas as pd
import logging
from main import dropnans
import matplotlib.pyplot as plt



class LSTMnet(nn.Module):
    def __init__(self, input_shape, hidden_size = 16, num_layers = 1, output_size = 1, optimizer = optim.Adam):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.optimizer = optimizer

        self.layer1 = nn.LSTM(input_size=input_shape[1], hidden_size= hidden_size, num_layers=num_layers, batch_first=True)
        self.layer2 = nn.Linear(hidden_size, output_size)


    def forward(self, input):
        #print(input.shape)
        #print(self.input_shape)

        assert input.shape == self.input_shape
        relu1 = F.relu(input)
        lstmd, (_, _) = self.layer1(relu1)
        # print(lstmd.shape)
        F.relu(lstmd, inplace=True)
        prediction = self.layer2(lstmd)
        # print(prediction)
        return prediction
def train(model, epochs: int, lr, batches):
        criterion = nn.MSELoss()
        optimizer = model.optimizer(model.parameters(), lr)
        running_loss = 0
        data = list()
        ts_train = time.perf_counter()
        for epoch in range(epochs):
           print("Epoch = ", epoch)
           ts = time.perf_counter()

           for i, (x, y) in enumerate(batches):
               #print(x.shape)
               x = x.reshape(model.input_shape)
               #print(x.shape)
               optimizer.zero_grad()
               out = model(x.float())
               #print("Loss from : " , out, y.type(torch.float32))
               loss = criterion(out.flatten(), y.type(torch.float32))

               loss.backward()
               optimizer.step()

               running_loss += loss.item()
               #if(loss.item() == torch.nan):
               #print(loss.item())
               #print(y[0])
               data.append({'update': i, 'epoch': epoch, 'loss': loss.item(), 'prediction': out.item()})
               #print(loss.item())
            #logging.info(f'Epoch took: {time.perf_counter() - ts:.2f}s')
        #logging.info(f'Finished training. {epochs} epochs took: {time.perf_counter() - ts_train:.2f}s')
        return data

def test(model, test_in, test_target):
    print("TEST in = ", test_in)
    #print(test_in.columns)

    metrics = list()

    for i, instance in enumerate(test_in):
        #print(instance)
        inst = instance.reshape(model.input_shape)
        #print(inst)
        output = model(inst.float()).item()
        error = abs(output - test_target[i].item())
        print(error)

        metrics.append({"error": error, "prediction": output, "target": test_target[i].item(), "id": int(test_in[i][0].item()), "day": int(test_in[i][1].item())})

    return metrics






def main():
    LSTM = LSTMnet(input_shape = (1,14)).float()

    print(LSTM)

    data = pd.read_csv("train.csv")

    test_data = pd.read_csv("test.csv")


    print(data.isnull().sum())
    morenans = ["office", "game", "social", "entertainment", "communication", "weather"]#["valence", "activity", "entertainment", "communication", "other", "mood", "social", "builtin", "screen","arousal"]
    data = dropnans(data, morenans)
    test_data = dropnans(test_data, morenans)

    # [id,day,screen,call,social,sms,builtin,utilities,arousal,finance,unknown,valence,office,activity,game,entertainment,weather,communication,travel,other,mood]
    print(data.isnull().sum())
    #print(data["screen"].shape)
    train_data = data.drop(labels = ["mood", "Unnamed: 0.1", "Unnamed: 0"], axis=1)
    test_in = test_data.drop(labels=["mood", "Unnamed: 0.1", "Unnamed: 0"], axis=1)
    print("TEST in = ", test_in)
    test_target = torch.from_numpy(np.array(test_data["mood"]))
    #train_data = dropnans(train_data, train_data.columns)
    print(train_data.dtypes)
    x = torch.tensor(train_data.values)
    test_in = torch.tensor(test_in.values)
    print("x.shape = ", x.shape)
    print(x.dtype)
    y = torch.from_numpy(np.array(data["mood"]))
    y = y[None, :].T

    print(y.shape)

    batches = []
    batches.extend(list(zip(x,y)))
    #print(batches)

    data = train(LSTM, 5, 0.01, batches)

    df = pd.DataFrame(data)
    df['epoch'] = df['epoch'] + 1

    plt.figure()
    df.groupby(by='epoch').mean()['loss'].plot()
    plt.ylabel('loss')
    plt.title("Basic 100 epochs")
    plt.show()

    metrics = test(LSTM, test_in, test_target)

    errordf = pd.DataFrame(metrics)
    errordf.to_csv("error_LSTM.csv")
    plt.figure()
    errordf["error"].plot()
    plt.show()
    print(errordf["error"].mean())


main()