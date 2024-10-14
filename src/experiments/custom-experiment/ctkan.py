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
import torch.nn.functional as F
from KANConv import KAN_Convolutional_Layer
from fastkan import FastKAN as KAN
from torch.nn.utils import weight_norm

from Transformer.test import output


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

    train_data, test_data = train_test_split(
        data, test_size=0.1, shuffle=True, random_state=42)
    train_dataset = TimeSeriesDataset(train_data, lookback)
    test_dataset = TimeSeriesDataset(test_data, lookback)
    train_dl = DataLoader(train_dataset, batch_size=32,
                          num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=32,
                         num_workers=8, pin_memory=True)

    # X_train = torch.from_numpy(X_train).float()
    # X_test = torch.from_numpy(X_test).float()
    # y_train = torch.from_numpy(y_train).float()
    # y_test = torch.from_numpy(y_test).float()

    return train_dl, test_dl


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = KAN_Convolutional_Layer(n_convs=n_outputs, kernel_size=(kernel_size, kernel_size), stride=stride,
                                             padding=padding, dilation=dilation)
        # self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
        #                        stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding[0])
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = KAN_Convolutional_Layer(n_convs=n_outputs, kernel_size=(kernel_size, kernel_size), stride=stride,
                                             padding=padding, dilation=dilation)
        # self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
        #                        stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding[0])
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.dropout1,
                                 self.conv2, self.chomp2, self.dropout2)
        # self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        x = self.net(x)  #
        return x


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            padding_size = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(out_channels, kernel_size, stride=(1, 1), dilation=(dilation_size, dilation_size),
                                     padding=(padding_size, padding_size), dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((num_channels[-1], 1))
        self.linear = KAN([528, num_inputs])

    def forward(self, x):
        x=self.network(x)
        x=x[:,:,:,0]
        x=x.reshape(x.size(0),-1)
        return self.linear(x).unsqueeze(1)


        # # print(f"x shape: {x.shape}")
        # x = self.network(x)
        # # print(f"x shape: {x.shape}")
        # x = x[:, :, :, 0]
        # # print(f"x shape: {x.shape}")
        # x=x.squeeze(2)
        # # print(f"x shape: {x.shape}")
        # x = self.adaptive_pool(x)
        # # print(f"x shape: {x.shape}")
        # x = x.squeeze(-1)
        # # print(f"x shape: {x.shape}")
        # return self.linear(x).unsqueeze(1)


class TemporalConvNetSimple(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNetSimple, self).__init__()
        self.conv = KAN_Convolutional_Layer(n_convs=num_channels[-1], kernel_size=(kernel_size, kernel_size))
        self.linear = KAN([num_channels[-1], num_inputs])

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, 0, 0]
        return self.linear(x).unsqueeze(1)


param_log = []


def check_model_params(model):
    for name, param in model.named_parameters():
        # if name=="network.3.conv2.weight_v":
        #     print(param)
        if torch.isnan(param).any():
            print(f"NaN detected in parameter {name}")
            plt.plot(param_log)
            return True
    return False


def train_model_large(num_inputs, train_dl, dropout):
    model = TemporalConvNet(num_inputs=num_inputs, num_channels=[2,2], dropout=dropout)
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
            train_dl, desc=f"Epoch {e + 1}/{num_epochs}", dynamic_ncols=True)
        for batch in progress_bar:
            X_train, y_train = batch
            X_train = X_train.transpose(1, 2).unsqueeze(1)
            y_train = y_train.unsqueeze(1)
            # print(f"X_train shape: {X_train.shape}. y_train shape: {y_train.shape}")
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_pred = model(X_train)
            # print(f"y_pred shape is: {y_pred.shape}\ty_train shape is: {y_train.shape}")
            loss = criterion(y_pred, y_train)
            check_model_params(model)
            # print(f"y_pred shape: {y_pred.shape}. y_train shape: {y_train.shape}")
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            optimiser.zero_grad()
            progress_bar.set_postfix(loss=loss.item())
    return model, loss.item()


def test_model(model, test_dl, train_loss, device):
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
    # file_path = '~/Documents/ML/EE/data/iTransformer_datasets/electricity/electricity.csv'
    # file_path = '~/Documents/ML/EE/data/iTransformer_datasets/traffic/traffic.csv'
    # file_path = '~/Documents/ML/EE/data/iTransformer_datasets/weather/weather.csv'

    base_dir = "/mnt/c/WorkDir/ComputerScience/MachineLearning/EE/src"
    file_path = base_dir+"/data/iTransformer_datasets/traffic/traffic.csv"
    data = load_data(file_path, category="traffic")

    scaled_data = scale_data(data)

    lookback = 96
    # X_train, X_test, y_train, y_test = get_train_test_data(
    #     scaled_data, lookback)
    train_dl, test_dl = get_train_test_data(scaled_data, lookback)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 输入的维度为1，只有Close收盘价
    input_dim = scaled_data.shape[1]
    num_epochs = 5
    learning_rate = 0.001
    weight_decay = 1e-4
    dropout = 0.1

    # Without KAN
    # model1, train_loss1 = train_model_small(input_dim, hidden_dim, num_layers,
    #                                         output_dim, num_heads, dropout,X_train,y_train kan=False)
    model1, train_loss1 = train_model_large(input_dim, train_dl, dropout)
    print("Without KAN:")
    # test_model(model1, X_test, y_test, train_loss1)
    test_model(model1, test_dl, train_loss1, device)
    # With KAN
    print("With KAN:")
    model2, train_loss2 = train_model_large(input_dim, test_dl, dropout)
    # test_model(model2, X_test, y_test, train_loss2)
    test_model(model2, test_dl, train_loss2, device)
