# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset
# import torch
# import sys
# import os
# # sys.path.append(os.path.abspath(os.path.join(
#     # '..', '..', 'MyModels', 'Transformer')))
# from ...MyModels.Transformer.Encoder import Encoder
# import torch.nn as nn
# from fastkan import FastKAN as KAN
# import torch.nn.functional as F


# def load_data(filepath):
#     data = pd.read_csv(filepath)
#     data = data.sort_values('Date')


# def scale_data(price):
#     # 进行不同的数据缩放，将数据缩放到-1和1之间
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1, 1))
#     return price


# def create_sequences(data, seq_length):
#     sequences = []
#     labels = []
#     for i in range(len(data) - seq_length):
#         seq = data[i:i + seq_length]
#         label = data[i + seq_length]
#         sequences.append(seq)
#         labels.append(label)
#     return np.array(sequences), np.array(labels)


# def get_train_test_data(price, lookback):
#     X, y = create_sequences(price[['Close']].values, lookback)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.1, shuffle=False, random_state=42)

#     train_dataset = TensorDataset(torch.from_numpy(
#         X_train).float(), torch.from_numpy(y_train).float())
#     test_dataset = TensorDataset(torch.from_numpy(
#         X_test).float(), torch.from_numpy(y_test).float())
#     # train_dl=DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=16,pin_memory=True)
#     # test_dl=DataLoader(test_dataset,batch_size=32,shuffle=False,num_workers=16,pin_memory=True)

#     X_train = torch.from_numpy(X_train).float()
#     X_test = torch.from_numpy(X_test).float()
#     y_train = torch.from_numpy(y_train).float()
#     y_test = torch.from_numpy(y_test).float()

#     return X_train, X_test, y_train, y_test


# class Transformer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_heads, dropout, kan=False):
#         super(Transformer, self).__init__()

#         # not using the nn transformer module
#         # self.encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=num_heads,dropout=dropout,batch_first=True)
#         # self.transformer_encoder=nn.TransformerEncoder(self.encoder_layer,num_layers=num_layers)
#         # self.fc=nn.Linear(hidden_dim,output_dim)

#         # using the using custom transformer module
#         self.transformer_encoder = Encoder(d_model=hidden_dim,
#                                            ffn_hidden=hidden_dim,
#                                            n_head=num_heads,
#                                            n_layers=num_layers,
#                                            drop_prob=dropout,
#                                            kan=kan)
#         if kan:
#             self.fc = KAN([hidden_dim, output_dim])
#         else:
#             self.fc = nn.Linear(hidden_dim, output_dim)

#         self.input_dim = input_dim
#         self.model_dim = hidden_dim
#         self.embedding = nn.Linear(input_dim, hidden_dim)

#     def forward(self, x):
#         x = self.embedding(x)*(self.model_dim**0.5)
#         x = self.transformer_encoder(x)
#         out = self.fc(x[:, -1, :])
#         return out


# def train_model(input_dim, hidden_dim, num_layers, output_dim, num_heads, dropout, kan=False):
#     model = Transformer(input_dim=input_dim,
#                         hidden_dim=hidden_dim,
#                         num_layers=num_layers,
#                         output_dim=output_dim,
#                         num_heads=num_heads,
#                         dropout=dropout,
#                         kan=True)
#     model.to(device)
#     criterion = torch.nn.MSELoss()
#     optimiser = torch.optim.Adam(
#         model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     # hist = np.zeros(num_epochs)
#     # lstm = []
#     lost_list = []

#     for t in range(num_epochs):
#         y_train_pred = model(X_train.to(device))

#         loss = criterion(y_train_pred, y_train.to(device))
#         print("Epoch ", t, "MSE: ", loss.item())
#         lost_list.append(loss.item())

#         optimiser.zero_grad()
#         loss.backward()
#         optimiser.step()

#     return model


# if __name__ == '__main__':
#     file_path = '../../data/600519.csv'
#     data = load_data(file_path)

#     price = scale_data(data[['Close']])

#     lookback = 20
#     X_train, X_test, y_train, y_test = get_train_test_data(price, lookback)
#     y_train_transformer = y_train
#     y_test_transformer = y_test

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # 输入的维度为1，只有Close收盘价
#     input_dim = 1
#     # 隐藏层特征的维度
#     hidden_dim = 10
#     # 循环的layers
#     num_layers = 1
#     # 预测后一天的收盘价
#     output_dim = 1
#     num_epochs = 300
#     learning_rate = 0.05
#     weight_decay = 1e-5
#     num_heads = 2
#     dropout = 0.1

#     model = train_model(input_dim, hidden_dim, num_layers,
#                         output_dim, num_heads, dropout, kan=True)

#     loss = nn.MSELoss()
#     # predict
#     y_test_pred = model(X_test.to(device))
#     # convert y_test to tensor
#     y_test = y_test.to(device)
#     # calculate MSE
#     print(loss(y_test_pred, y_test))
