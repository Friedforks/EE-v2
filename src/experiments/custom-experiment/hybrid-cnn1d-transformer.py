import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpmath import floor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import torch
from tqdm import tqdm
from Transformer.Encoder import Encoder
import torch.nn as nn
import torch.nn.functional as F
import config

def load_data(filepath, category):
    data = pd.read_csv(filepath)
    if 'date' in data.columns:
        data.drop(['date'], axis=1, inplace=True)
    if 'Date' in data.columns:
        data.drop(['Date'], axis=1, inplace=True)
    if 'Adj Close' in data.columns:
        data.drop(['Adj Close'], axis=1, inplace=True)
    if 'Code' in data.columns:
        data.drop(['Code'], axis=1, inplace=True)
    if 'Time' in data.columns:
        data.drop(['Time'], axis=1, inplace=True)
    return data.dropna()


def scale_data(data):
    data = data.copy()
    # 进行不同的数据缩放，将数据缩放到-1和1之间
    scaler = MinMaxScaler(feature_range=(0, 1))
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
    train_data, test_data = train_test_split(
        data, test_size=0.1, shuffle=True, random_state=42)
    train_dataset = TimeSeriesDataset(train_data, lookback)
    test_dataset = TimeSeriesDataset(test_data, lookback)
    train_dl = DataLoader(train_dataset, batch_size=128,
                          num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=128,
                         num_workers=8, pin_memory=True)

    return train_dl, test_dl


class TemporalConvTransformer(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim, n_head, transformer_layers, seq_len, kernel_size, stride,
                 dropout=0.2):
        super(TemporalConvTransformer, self).__init__()
        self.conv = nn.Conv1d(in_channels=num_outputs,
                              out_channels=num_outputs,
                              kernel_size=kernel_size,
                              stride=stride
                              )
        # self.pool1=nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        # self.conv2=nn.Conv2d(in_channels=16,out_channels=1,kernel_size=(3,3),stride=1,padding=0)
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.transformer_encoder = Encoder(d_model=hidden_dim, n_head=n_head, n_layers=transformer_layers,
                                           ffn_hidden=hidden_dim, drop_prob=dropout, kan=False)
        self.linear = nn.Linear(hidden_dim, num_outputs)
        self.split_point = int(0.8 * seq_len)

    def forward(self, x):
        # print(f"x shape: {x.shape}")
        batch_size, _, num_features, seq_len = x.shape
        cnn_input = x[:, :, :, :self.split_point].squeeze(1)
        # print(f"cnn_input shape: {x.shape}")
        transformer_input = x[:, :, :, self.split_point:].squeeze(1)
        # print(f"transformer input shape: {transformer_input.shape}")
        cnn_output = self.conv(cnn_input)
        # print(f"cnn output shape: {cnn_output.shape}")
        cnn_output = self.relu(cnn_output)
        # x=self.pool1(x)
        # x=self.conv2(x)
        # x=self.relu(x)
        # x=self.pool2(x)
        # [batch_size,1,features,num_outputs]
        # [32,1,10,47]
        # print(f"x shape: {x.shape}")

        # combined [batch_size,1,features,num_outputs]
        combined = torch.cat([cnn_output, transformer_input], dim=2)
        # [batch_size,features,seq_len]
        self.transformer_encoder(combined)
        # print(f"x shape: {x.shape}")
        # print(f"x shape: {x.shape}")
        combined = self.linear(combined[:, 0, :]).unsqueeze(1)
        return combined


loss_log = []


def train_model_large(num_inputs, train_dl, hidden_dim, n_head, transformer_layers, dropout, lookback, kernel_size, stride):
    model = TemporalConvTransformer(num_inputs=num_inputs,
                                    hidden_dim=hidden_dim,
                                    n_head=n_head,
                                    transformer_layers=transformer_layers,
                                    dropout=dropout,
                                    num_outputs=num_inputs,
                                    seq_len=lookback,
                                    kernel_size=kernel_size,
                                    stride=stride
                                    )
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for e in range(num_epochs):
        print(f"Training the Epoch: {e + 1}")

        progress_bar = tqdm(
            train_dl, desc=f"Epoch {e + 1}/{num_epochs}", dynamic_ncols=True)
        for batch in progress_bar:
            X_train, y_train = batch
            X_train = X_train.transpose(1, 2).unsqueeze(1)
            y_train = y_train.unsqueeze(1)
            # print(f"X_train shape: {X_train.shape}. y_train shape: {y_train.shape}")
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_pred = model(X_train)
            # print("y_pred shape: ", y_pred.shape)
            # print("y_train shape: ", y_train.shape)
            # print(f"y_pred shape is: {y_pred.shape}\ty_train shape is: {y_train.shape}")
            loss = criterion(y_pred, y_train)
            # print(f"y_pred shape: {y_pred.shape}. y_train shape: {y_train.shape}")
            loss.backward()
            loss_log.append(loss.item())
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            optimiser.zero_grad()
            progress_bar.set_postfix(loss=loss.item())
    return model, loss.item()


def test_model(model, test_dl, train_loss, device):
    loss_hist = []
    for batch in test_dl:
        X_test, y_test = batch
        X_test = X_test.transpose(1, 2).unsqueeze(1)
        y_test = y_test.unsqueeze(1)
        loss = nn.MSELoss()
        # predict
        y_test_pred = model(X_test.to(device))
        # convert y_test to tensor
        y_test = y_test.to(device)
        loss_hist.append(loss(y_test_pred, y_test).item())
    # calculate MSE
    print(f"Train loss:{train_loss}\nTest loss: {np.mean(loss_hist)}")
    return np.mean(loss_hist)


if __name__ == '__main__':
    base_dir = config.base_dir
    file_path = base_dir+"/data/iTransformer_datasets/traffic/traffic.csv"
    data = load_data(file_path, category="traffic")

    scaled_data = scale_data(data)

    # lookback = [128, 256, 512, 1024]
    seq_len=128
    train_loss_log = []
    test_loss_log = []
    train_dl, test_dl = get_train_test_data(scaled_data, seq_len)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = scaled_data.shape[1]
    output_dim = scaled_data.shape[1]
    num_epochs = config.epochs
    kernel_size = 30
    stride = 5

    hidden_dim = int((int(0.8*seq_len)-kernel_size) /
                     stride+1+seq_len-(int(0.8*seq_len)))

    print(
        f"Hidden dim: {hidden_dim} on sequence length: {seq_len} kernel size: {kernel_size} stride: {stride}")
    n_head = config.num_heads
    transformer_layers = config.transformer_layers
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    dropout = config.dropout

    # Without KAN
    # model1, train_loss1 = train_model_small(input_dim, hidden_dim, num_layers,
    #                                         output_dim, num_heads, dropout,X_train,y_train kan=False)
    model1, train_loss1 = train_model_large(input_dim,
                                            train_dl,
                                            hidden_dim,
                                            n_head,
                                            transformer_layers,
                                            dropout,
                                            seq_len,
                                            kernel_size,
                                            stride
                                            )
    train_loss_log.append(train_loss1)
    # test_model(model1, X_test, y_test, train_loss1)
    test_loss_1 = test_model(model1, test_dl, train_loss1, device)
    test_loss_log.append(test_loss_1)

    # save the loss log
    print("Train loss log: ", train_loss_log)
    np.save(base_dir+"/experiments/loss-data/traffic/TCformer_1d_train_loss_log.npy", train_loss_log)
    np.save(base_dir+"/experiments/loss-data/traffic/TCformer_1d_test_loss_log.npy", test_loss_log)
