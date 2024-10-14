import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import torch
from tqdm import tqdm
from Encoder import Encoder
import torch.nn as nn
from fastkan import FastKAN as KAN
import torch.nn.functional as F


def load_data(filepath, category):
    data = pd.read_csv(filepath)
    data.drop(['date'], axis=1, inplace=True)
    return data


def scale_data(data):
    data = data.copy()
    # 进行不同的数据缩放，将数据缩放到-1和1之间
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    return data


def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length, :]
        label = data[i + seq_length, :]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


class TimeSeriesDataset(IterableDataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __iter__(self):
        for i in range(len(self.data) - self.seq_length):
            seq = self.data[i:i + self.seq_length]
            label = self.data[i + self.seq_length]
            yield torch.from_numpy(seq).float(), torch.from_numpy(label).float()


def get_train_test_data(data, lookback):
    # X, y = create_sequences(data, lookback)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.1, shuffle=True, random_state=42)

    # train_dataset = TensorDataset(torch.from_numpy(
    #     X_train).float(), torch.from_numpy(y_train).float())
    # test_dataset = TensorDataset(torch.from_numpy(
    #     X_test).float(), torch.from_numpy(y_test).float())

    train_data,test_data=train_test_split(data,test_size=0.1,shuffle=True,random_state=42)
    train_dataset=TimeSeriesDataset(train_data,lookback)
    test_dataset=TimeSeriesDataset(test_data,lookback)
    train_dl = DataLoader(train_dataset, batch_size=32,
                          num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=32,
                         num_workers=8, pin_memory=True)

    # X_train = torch.from_numpy(X_train).float()
    # X_test = torch.from_numpy(X_test).float()
    # y_train = torch.from_numpy(y_train).float()
    # y_test = torch.from_numpy(y_test).float()

    return train_dl, test_dl


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_heads, dropout, kan=False):
        super(Transformer, self).__init__()

        # not using the nn transformer module
        # self.encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=num_heads,dropout=dropout,batch_first=True)
        # self.transformer_encoder=nn.TransformerEncoder(self.encoder_layer,num_layers=num_layers)
        # self.fc=nn.Linear(hidden_dim,output_dim)

        # using the using custom transformer module
        self.transformer_encoder = Encoder(d_model=hidden_dim,
                                           ffn_hidden=hidden_dim,
                                           n_head=num_heads,
                                           n_layers=num_layers,
                                           drop_prob=dropout,
                                           kan=kan)
        if kan:
            self.fc = KAN([hidden_dim, output_dim])
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

        self.input_dim = input_dim
        self.model_dim = hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.embedding(x)*(self.model_dim**0.5)
        x = self.transformer_encoder(x)
        out = self.fc(x[:, -1, :])
        return out


def train_model_small(input_dim, hidden_dim, num_layers, output_dim, num_heads, dropout, X_train, y_train, kan=False):
    model = Transformer(input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        output_dim=output_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        kan=kan)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # hist = np.zeros(num_epochs)
    # lstm = []
    lost_list = []

    for t in range(num_epochs):
        y_train_pred = model(X_train.to(device))

        loss = criterion(y_train_pred, y_train.to(device))
        # print("Epoch ", t, "MSE: ", loss.item())
        lost_list.append(loss.item())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return model, loss.item()


def train_model_large(input_dim, hidden_dim, num_layers, output_dim, num_heads, dropout, train_dl, kan=False):
    model = Transformer(input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        output_dim=output_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        kan=kan)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # hist = np.zeros(num_epochs)
    # lstm = []
    lost_list = []

    for e in range(num_epochs):
        print(f"Training the Epoch: {e + 1}")

        progress_bar = tqdm(
            train_dl, desc=f"Epoch {e+1}/{num_epochs}", dynamic_ncols=True)
        for batch in progress_bar:
            X_train, y_train = batch
            # print(f"X_train shape: {X_train.shape}. y_train shape: {y_train.shape}")
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            # print(f"y_pred shape: {y_pred.shape}. y_train shape: {y_train.shape}")
            loss.backward()

            optimiser.step()
            optimiser.zero_grad()
            progress_bar.set_postfix(loss=loss.item())
    return model, loss.item()


def test_model(model, test_dl, train_loss):
    loss_hist = []
    for batch in test_dl:
        X_test, y_test = batch
        loss = nn.MSELoss()
        # predict
        y_test_pred = model(X_test.to(device))
        # convert y_test to tensor
        y_test = y_test.to(device)
        loss_hist.append(loss(y_test_pred, y_test).item())
    # calculate MSE
    print(f"Train loss:{train_loss}\nTest loss: {np.mean(loss_hist)}")


if __name__ == '__main__':
    file_path = '~/Documents/ML/EE/data/iTransformer_datasets/electricity/electricity.csv'
    # file_path = '~/Documents/ML/EE/data/iTransformer_datasets/traffic/traffic.csv'
    # file_path = '~/Documents/ML/EE/data/iTransformer_datasets/weather/weather.csv'
    data = load_data(file_path, category="traffic")
    scaled_data = scale_data(data)

    lookback = 96
    # X_train, X_test, y_train, y_test = get_train_test_data(
    #     scaled_data, lookback)
    train_dl, test_dl = get_train_test_data(scaled_data, lookback)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 输入的维度为1，只有Close收盘价
    input_dim = scaled_data.shape[1]
    # 隐藏层特征的维度
    hidden_dim = 16
    # 循环的layers
    num_layers = 10
    # 预测后一天的收盘价
    output_dim = scaled_data.shape[1]
    num_epochs = 5
    learning_rate = 0.001
    weight_decay = 1e-5
    num_heads = 2
    dropout = 0.1

    # Withou KAN
    # model1, train_loss1 = train_model_small(input_dim, hidden_dim, num_layers,
    #                                         output_dim, num_heads, dropout,X_train,y_train kan=False)
    model1, train_loss1 = train_model_large(input_dim, hidden_dim, num_layers,
                                            output_dim, num_heads, dropout, train_dl, kan=False)
    print("Without KAN:")
    # test_model(model1, X_test, y_test, train_loss1)
    test_model(model1, test_dl, train_loss1)
    # With KAN
    print("With KAN:")
    model2, train_loss2 = train_model_large(input_dim, hidden_dim, num_layers,
                                            output_dim, num_heads, dropout, train_dl, kan=True)
    # test_model(model2, X_test, y_test, train_loss2)
    test_model(model2, test_dl, train_loss2)
