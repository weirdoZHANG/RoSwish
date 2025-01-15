import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nonlinearity_name = sys.argv[1] if len(sys.argv) > 1 else 'roswish'  # 'relu', 'gelu', 'silu', 'selu', 'mish', 'prelu', 'erfact', 'swish', 'psgu', or 'roswish'
learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3  # 1e-3, 1e-4, 1e-5
p = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0  # 0.0, 0.5

training_epochs = 50
batch_size = 32
sequence_length = 96
pred_length = 96

n_hidden = 64
n_features = 7


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
        self.dropout = nn.Dropout(p)

        self.a = nn.Parameter(torch.full((1, hidden_size), 0.75))
        self.b = nn.Parameter(torch.full((1, hidden_size), 0.75))

        self.c = nn.Parameter(torch.full((1, hidden_size), 1.00))

        self.alpha = nn.Parameter(torch.full((1, hidden_size), 0.817))
        self.beta = nn.Parameter(torch.full((1, hidden_size), 3.000))

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
        elif nonlinearity == 'roswish':
            return self.RoSwish
        else:
            raise ValueError("Unsupported nonlinearity: choose 'relu', 'gelu', 'silu', 'selu', 'mish',"
                             " 'prelu', 'erfact', 'swish', 'psgu', or 'roswish' for nonlinearity_name")

    def ErfAct(self, x):
        return x * torch.erf(self.a * torch.exp(self.b * x))

    def Swish(self, x):
        return x * torch.sigmoid(self.c * x)

    def PSGU(self, x):
        return x * torch.tanh(self.beta * torch.sigmoid(x))

    def RoSwish(self, x):
        return (x + self.alpha) * torch.sigmoid(x * self.beta) - self.alpha / 2

    def forward(self, x, is_training):
        for layer in self.layers[:-1]:
            x = self.nonlinearity(layer(x))
            if is_training:
                x = self.dropout(x)

        x = self.layers[-1](x)
        x = x[:, -pred_length:, :]
        return x


df_data = pd.read_csv('data/ETTm2/ETTm2.csv')
cols = list(df_data.columns)
cols.remove('date')
df_data = df_data[cols]
df_train = df_data.sample(frac=0.6)
scaler = StandardScaler()
scaler.fit(df_train.values)
data = scaler.transform(df_data.values)

dataset = TimeSeriesDataset(data, sequence_length, pred_length)
train_size = int(len(dataset) * 0.6)
val_size = int(len(dataset) * 0.2)
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

if os.path.exists("./data/ettm2_pred_" + nonlinearity_name + ".p"):
    history = pickle.load(open("./data/ettm2_pred_" + nonlinearity_name + ".p", "rb"))
    key_str = str(len(history)//6 + 1)
    history["lr" + key_str] = learning_rate
    history["dropout" + key_str] = p
    history["train_loss" + key_str] = []
    history["val_loss" + key_str] = []
    history["test_loss" + key_str] = []
    history["test_mae" + key_str] = []
else:
    history = {
        "lr1": learning_rate, "dropout1": p,
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
        outputs = model(bx.float(), is_training=True)
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
            outputs = model(bx.float(), is_training=False)
            val_loss = criterion(outputs, by.float())
            val_losses.append(val_loss.item())
        history["val_loss" + key_str].append(np.mean(val_losses))

    with torch.no_grad():
        test_losses = []
        test_maes = []
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx.float(), is_training=False)
            test_loss = criterion(outputs, by.float())
            test_mae = mae(outputs, by.float())
            test_losses.append(test_loss.item())
            test_maes.append(test_mae.item())
        history["test_loss" + key_str].append(np.mean(test_losses))
        history["test_mae" + key_str].append(np.mean(test_maes))

    print(f'Epoch [{epoch + 1}/{training_epochs}], Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}, '
          f'Test Loss: {np.mean(test_losses):.4f}, Test MAE: {np.mean(test_maes):.4f}')

pickle.dump(history, open("./data/ettm2_pred_" + nonlinearity_name + ".p", "wb"))
