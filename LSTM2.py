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
    def __init__(self, input_shape, hidden_size = 16, num_layers = 1, output_size = 2, optimizer = optim.Adam):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.optimizer = optimizer

        self.layer1 = nn.LSTM(input_size=input_shape[1], hidden_size= hidden_size, num_layers=num_layers, batch_first=True, dropout= 0.2)

        self.layer2 = nn.Linear(hidden_size, output_size)


    def forward(self, input):
        #print(input.shape)
        #print(self.input_shape)

        assert input.shape == self.input_shape
        relu1 = F.relu(input)
        lstmd, (_,_) = self.layer1(relu1)
        #print(lstmd.shape)
        F.relu(lstmd, inplace= True)
        prediction = self.layer2(lstmd)
        #print(prediction)
        return prediction
def train(model, epochs: int, lr, batches):
        criterion = nn.L1Loss()
        optimizer = model.optimizer(model.parameters(), lr)
        running_loss = 0
        data = list()
        ts_train = time.perf_counter()
        for epoch in range(epochs):
           ts = time.perf_counter()

           for i, (x, y) in enumerate(batches):
               #print(x.shape)
               x = x.reshape(model.input_shape)
               #print(x.shape)
               optimizer.zero_grad()
               out = model(x.float())
               loss = criterion(out.flatten(), y.type(torch.float32))

               loss.backward()
               optimizer.step()

               running_loss += loss.item()
               #if(loss.item() == torch.nan):
               #print(loss.item())
               #print(y[0])
               data.append({'update': i, 'epoch': epoch, 'loss': loss.item()})
               #print(loss.item())
            #logging.info(f'Epoch took: {time.perf_counter() - ts:.2f}s')
        #logging.info(f'Finished training. {epochs} epochs took: {time.perf_counter() - ts_train:.2f}s')
        return data

def main():
    LSTM = LSTMnet(input_shape = (1,18)).float()

    print(LSTM)

    data = pd.read_csv("shifted.csv")

    print(data.isnull().sum())
    #morenans = ["valence", "activity", "entertainment", "communication", "other", "mood", "social", "builtin", "screen","arousal"]
    #data = dropnans(data, morenans)

    # [id,day,screen,call,social,sms,builtin,utilities,arousal,finance,unknown,valence,office,activity,game,entertainment,weather,communication,travel,other,mood]
    print(data.isnull().sum())
    #print(data["screen"].shape)
    train_data = data.drop(labels = ["id","day", "mood", "Unnamed: 0.1", "Unnamed: 0"], axis=1)
    train_data = dropnans(train_data, train_data.columns)
    print(train_data.dtypes)
    x = torch.tensor(train_data.values)
    print("x.shape = ", x.shape)
    print(x.dtype)
    y = torch.from_numpy(np.array(data["mood"]))
    y = y[None, :].T

    print(y.shape)

    batches = []
    batches.extend(list(zip(x,y)))
    #print(batches)

    data = train(LSTM, 30, 0.05, batches)

    df = pd.DataFrame(data)
    df['epoch'] = df['epoch'] + 1

    plt.figure()
    df.groupby(by='epoch').mean()['loss'].plot()
    plt.ylabel('loss')
    plt.title("dropout = 0.2, lr = 0.02, relu")
    plt.show()


main()