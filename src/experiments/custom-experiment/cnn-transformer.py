import numpy as np
import pandas as pd
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


def load_data(filepath, category):
    data = pd.read_csv(filepath)
    data.drop(['date'], axis=1, inplace=True)
    return data


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
    train_dl = DataLoader(train_dataset, batch_size=512,
                          num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=512,
                         num_workers=8, pin_memory=True)

    return train_dl, test_dl


class TemporalConvTransformer(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim, n_head, transformer_layers,
                 dropout=0.2):
        super(TemporalConvTransformer, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(8,8), stride=1, padding=3)
        self.pool1=nn.MaxPool2d(kernel_size=2)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(in_channels=16,out_channels=1,kernel_size=(3,3),stride=1,padding=0)
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.transformer_encoder = Encoder(d_model=hidden_dim, n_head=n_head, n_layers=transformer_layers,
                                           ffn_hidden=hidden_dim, drop_prob=dropout, kan=False)
        self.linear=nn.Linear(hidden_dim,num_outputs)
    def forward(self, x):
        # [batch_size,1,features,num_inputs]
        # [32,1,21,96]
        # print(f"x shape: {x.shape}")
        x = self.conv(x)
        x=self.relu(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.relu(x)
        # x=self.pool2(x)
        # [batch_size,1,features,num_outputs]
        # [32,1,10,47]
        # print(f"x shape: {x.shape}")
        x=x.squeeze(1).transpose(1,2)

        # [batch_size,features,num_outputs]
        self.transformer_encoder(x)
        # print(f"x shape: {x.shape}")
        # print(f"x shape: {x.shape}")
        x = self.linear(x[:,0,:]).unsqueeze(1)
        return x


def check_model_params(model):
    for name, param in model.named_parameters():
        # if name=="network.3.conv2.weight_v":
        #     print(param)
        if torch.isnan(param).any():
            print(f"NaN detected in parameter {name}")
            print(f"loss log: {loss_log}")
            return True
    return False


loss_log = []


def train_model_large(num_inputs, train_dl, hidden_dim, n_head, transformer_layers, dropout):
    model = TemporalConvTransformer(num_inputs=num_inputs,
                                    hidden_dim=hidden_dim,
                                    n_head=n_head,
                                    transformer_layers=transformer_layers,
                                    dropout=dropout,
                                    num_outputs=num_inputs,
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
            print("y_pred shape: ", y_pred.shape)
            print("y_train shape: ", y_train.shape)
            # print(f"y_pred shape is: {y_pred.shape}\ty_train shape is: {y_train.shape}")
            loss = criterion(y_pred, y_train)
            check_model_params(model)
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
    # file_path = '~/Documents/ML/EE/data/iTransformer_datasets/electricity/electricity.csv'
    # file_path = '~/Documents/ML/EE/data/iTransformer_datasets/traffic/traffic.csv'
    file_path = '~/Documents/ML/EE/data/iTransformer_datasets/weather/weather.csv'
    data = load_data(file_path, category="traffic")

    scaled_data = scale_data(data)

    lookback = [64, 128, 256, 512,1024]
    train_loss_log=[]
    test_loss_log=[]
    for i in lookback:
        # X_train, X_test, y_train, y_test = get_train_test_data(
        #     scaled_data, lookback)
        train_dl, test_dl = get_train_test_data(scaled_data, i)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 输入的维度为1，只有Close收盘价
        input_dim = scaled_data.shape[1]
        output_dim = scaled_data.shape[1]
        num_epochs = 3
        hidden_dim = 8
        n_head = 2
        transformer_layers = 4
        learning_rate = 0.001
        weight_decay = 1e-4
        dropout = 0.1

        # Without KAN
        # model1, train_loss1 = train_model_small(input_dim, hidden_dim, num_layers,
        #                                         output_dim, num_heads, dropout,X_train,y_train kan=False)
        model1, train_loss1 = train_model_large(input_dim,
                                                train_dl,
                                                hidden_dim,
                                                n_head,
                                                transformer_layers,
                                                dropout,

                                                )
        train_loss_log.append(train_loss1)
        # test_model(model1, X_test, y_test, train_loss1)
        test_loss_1=test_model(model1, test_dl, train_loss1, device)
        test_loss_log.append(test_loss_1)

    # save the loss log
    np.save("ctransformer_train_loss_log.npy",train_loss_log)
    np.save("ctransformer_test_loss_log.npy",test_loss_log)
