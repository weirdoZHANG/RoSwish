import os
import sys
import pickle
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# relu, gelu, silu, selu, mish, prelu, erfact, swish, psgu, aoaf, sinlu, tanhlu, aqulu, or roswish
nonlinearity_name = sys.argv[1] if len(sys.argv) > 1 else 'roswish'
isNsSin = sys.argv[2] if len(sys.argv) > 2 else True

training_epochs = 100
batch_size = 32
sequence_length = 96
pred_length = 96

n_hidden = 256
n_features = 2
learning_rate = 1e-3


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_length):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length]
        return x, y


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, n_features, nonlinearity_name):
        super(FullyConnectedNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(7):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, n_features))

        self.a_NsSin = nn.Parameter(torch.full((1, n_hidden), 1.0))
        self.b_NsSin = nn.Parameter(torch.full((1, n_hidden), 0.0))

        # ErfAct
        self.ErfAct_alpha = nn.Parameter(torch.full((1, n_hidden), 0.75))
        self.ErfAct_beta = nn.Parameter(torch.full((1, n_hidden), 0.75))

        # Swish
        self.Swish_alpha = nn.Parameter(torch.full((1, n_hidden), 1.0))

        # PSGU
        self.PSGU_alpha = nn.Parameter(torch.full((1, n_hidden), 3.0))

        # SinLU
        self.SinLU_alpha = nn.Parameter(torch.full((1, n_hidden), 1.0))
        self.SinLU_beta = nn.Parameter(torch.full((1, n_hidden), 1.0))

        # tanhLU
        self.tanhLU_alpha = nn.Parameter(torch.full((1, n_hidden), 1.0))
        self.tanhLU_beta = nn.Parameter(torch.full((1, n_hidden), 0.0))
        self.tanhLU_gamma = nn.Parameter(torch.full((1, n_hidden), 1.0))

        # AQuLU
        self.AQuLU_alpha = nn.Parameter(torch.full((1, n_hidden), 7 / 30))
        self.AQuLU_beta = nn.Parameter(torch.full((1, n_hidden), math.sqrt(1 / 2)))

        # RoSwish
        self.alpha = nn.Parameter(torch.full((1, n_hidden), 0.817))
        self.beta = nn.Parameter(torch.full((1, n_hidden), 3.000))

        self.nonlinearity = self.get_nonlinearity(nonlinearity_name)

    def get_nonlinearity(self, nonlinearity):
        if nonlinearity == 'relu':
            return nn.ReLU()
        elif nonlinearity == 'gelu':
            return nn.GELU()
        elif nonlinearity == 'silu':
            return nn.SiLU()
        elif nonlinearity == 'selu':
            return nn.SELU()
        elif nonlinearity == 'mish':
            return nn.Mish()
        elif nonlinearity == 'prelu':
            return nn.PReLU()
        elif nonlinearity == 'erfact':
            return self.ErfAct
        elif nonlinearity == 'swish':
            return self.Swish
        elif nonlinearity == 'psgu':
            return self.PSGU
        elif nonlinearity == 'aoaf':
            return self.AOAF
        elif nonlinearity == 'sinlu':
            return self.SinLU
        elif nonlinearity == 'tanhlu':
            return self.tanhLU
        elif nonlinearity == 'aqulu':
            return self.AQuLU
        elif nonlinearity == 'roswish':
            return self.RoSwish
        else:
            raise ValueError("Unsupported nonlinearity: choose 'relu, gelu, silu, selu, mish, prelu, erfact, swish, "
                             "psgu, aoaf, sinlu, tanhlu, aqulu, or roswish' for nonlinearity_name")

    def ErfAct(self, x):
        return x * torch.erf(self.ErfAct_alpha * torch.exp(self.ErfAct_beta * x))

    def Swish(self, x):
        return x * torch.sigmoid(self.Swish_alpha * x)

    def PSGU(self, x):
        return x * torch.tanh(self.PSGU_alpha * torch.sigmoid(x))

    def AOAF(self, x):
        mean = torch.mean(x)
        return torch.where(x > 0.17 * mean, x - 0.17 * mean, torch.tensor(0.0, device=x.device)) + 0.17 * mean

    def SinLU(self, x):
        return (x + self.SinLU_alpha * torch.sin(self.SinLU_beta * x)) * torch.sigmoid(x)

    def tanhLU(self, x):
        return self.tanhLU_alpha * torch.tanh(self.tanhLU_gamma * x) + self.tanhLU_beta * x

    def AQuLU(self, x):
        return torch.where(
            x >= (1 - self.AQuLU_beta) / self.AQuLU_alpha,
            x,
            torch.where(
                (x >= -self.AQuLU_beta / self.AQuLU_alpha) & (x < (1 - self.AQuLU_beta) / self.AQuLU_alpha),
                (x ** 2) * self.AQuLU_alpha + x * self.AQuLU_beta,
                torch.tensor(0.0, device=x.device)
            )
        )

    def RoSwish(self, x):
        return (x + self.alpha) * torch.sigmoid(x * self.beta) - self.alpha / 2

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            f = self.a_NsSin * x + self.b_NsSin * torch.sin(x) if isNsSin else x
            x = self.nonlinearity(f)
        x = self.layers[-1](x)
        return x[:, -pred_length:, :]


df_data = pd.read_csv('data/Exchange/exchange_rate.csv')
cols = ['6', 'OT']
df_data = df_data[cols]
df_data = df_data.tail(730)
print("exchange_rate shape: ", df_data.shape)
df_train = df_data.sample(frac=0.7)
scaler = StandardScaler()
scaler.fit(df_train.values)
data = scaler.transform(df_data.values)

dataset = TimeSeriesDataset(data, sequence_length, pred_length)
train_size = int(len(dataset) * 0.7)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = FullyConnectedNetwork(n_features, n_hidden, n_features, nonlinearity_name).to(device)
criterion = nn.MSELoss()
mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


if not os.path.exists("./data/"):
    os.makedirs("./data/")

if os.path.exists("./data/exchange_rate_pred_" + nonlinearity_name + ".p"):
    history = pickle.load(open("./data/exchange_rate_pred_" + nonlinearity_name + ".p", "rb"))
    key_str = str(len(history)//5 + 1)
    history["lr" + key_str] = learning_rate
    history["train_loss" + key_str] = []
    history["val_loss" + key_str] = []
    history["test_loss" + key_str] = []
    history["test_mae" + key_str] = []
else:
    history = {
        "lr1": learning_rate,
        'train_loss1': [], 'val_loss1': [],
        'test_loss1': [], 'test_mae1': []
    }
    key_str = '1'


model.train()
for epoch in range(training_epochs):
    train_losses = []
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        outputs = model(bx.float())
        loss = criterion(outputs, by.float())
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    history["train_loss" + key_str].append(np.mean(train_losses))

    model.eval()
    with torch.no_grad():
        val_losses = []
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx.float())
            val_loss = criterion(outputs, by.float())
            val_losses.append(val_loss.item())
        history["val_loss" + key_str].append(np.mean(val_losses))

    with torch.no_grad():
        test_losses = []
        test_maes = []
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx.float())
            test_loss = criterion(outputs, by.float())
            test_mae = mae(outputs, by.float())
            test_losses.append(test_loss.item())
            test_maes.append(test_mae.item())
        history["test_loss" + key_str].append(np.mean(test_losses))
        history["test_mae" + key_str].append(np.mean(test_maes))

    print(f'Epoch [{epoch + 1}/{training_epochs}], Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}, '
          f'Test Loss: {np.mean(test_losses):.4f}, Test MAE: {np.mean(test_maes):.4f}')

pickle.dump(history, open("./data/exchange_rate_pred_" + nonlinearity_name + ".p", "wb"))
