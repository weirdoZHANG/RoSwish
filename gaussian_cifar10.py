import os
import sys
import pickle
import argparse
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from numpy.linalg import svd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# relu, gelu, silu, selu, mish, prelu, erfact, swish, psgu, aoaf, sinlu, tanhlu, aqulu, or roswish
activation_name = sys.argv[1] if len(sys.argv) > 1 else 'roswish'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
args = parser.parse_args()

seed = random.randint(1, 2 ** 15)
print("seed: ", seed)
rng = np.random.RandomState(seed)
torch.manual_seed(seed)

exp_dir = f"./data/{activation_name}_{args.batch_size}_{args.learning_rate}"
os.makedirs(exp_dir, exist_ok=True)

results_file = open(f"{exp_dir}/results.csv", 'w')
results_file.write('epoch, train_loss, train_err, test_loss, test_err\n')
results_file.flush()


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='latin1')
    return {
        'x': np.asarray((-127.5 + d['data'].reshape((10000, 3, 32, 32))) / 128., dtype=np.float32),
        'y': np.array(d['labels']).astype(np.uint8)
    }


train_data = [unpickle(f'./cifar-10-batches-py/data_batch_{i}') for i in range(1, 6)]
trainx = np.concatenate([d['x'] for d in train_data], axis=0)
trainy = np.concatenate([d['y'] for d in train_data])
test_data = unpickle('./cifar-10-batches-py/test_batch')
testx = test_data['x']
testy = test_data['y']


class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        self.ZCA_mat = None
        self.inv_ZCA_mat = None
        self.mean = None
        if x is not None:
            self.fit(x)

    def fit(self, x):
        x_flat = x.reshape(x.shape[0], -1)
        self.mean = np.mean(x_flat, axis=0)
        x_centered = x_flat - self.mean
        sigma = np.dot(x_centered.T, x_centered) / x_centered.shape[0]
        U, S, Vt = svd(sigma, full_matrices=False)

        self.ZCA_mat = np.einsum('ij,j,jk->ik', U, 1.0 / np.sqrt(S + self.regularization), U.T)
        self.inv_ZCA_mat = np.einsum('ij,j,jk->ik', U, np.sqrt(S + self.regularization), U.T)

    def apply(self, x):
        x_flat = x.reshape(x.shape[0], -1) - self.mean
        x_whitened = np.dot(x_flat, self.ZCA_mat)
        return x_whitened.reshape(x.shape)

    def invert(self, x):
        x_flat = x.reshape(x.shape[0], -1)
        x_original = np.dot(x_flat, self.inv_ZCA_mat) + self.mean
        return x_original.reshape(x.shape)


whitener = ZCA(x=trainx)
trainx_white = whitener.apply(trainx)
testx_white = whitener.apply(testx)

inds = rng.permutation(trainx_white.shape[0])
trainx_white = trainx_white[inds]
trainy = trainy[inds]


class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


class ErfAct(nn.Module):
    def __init__(self, n_hidden):
        super(ErfAct, self).__init__()
        self.ErfAct_alpha = nn.Parameter(torch.full((1, n_hidden, 1, 1), 0.75))
        self.ErfAct_beta = nn.Parameter(torch.full((1, n_hidden, 1, 1), 0.75))

    def forward(self, x):
        return x * torch.erf(self.ErfAct_alpha * torch.exp(self.ErfAct_beta * x))


class Swish(nn.Module):
    def __init__(self, n_hidden):
        super(Swish, self).__init__()
        self.Swish_alpha = nn.Parameter(torch.full((1, n_hidden, 1, 1), 1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.Swish_alpha * x)


class PSGU(nn.Module):
    def __init__(self, n_hidden):
        super(PSGU, self).__init__()
        self.PSGU_alpha = nn.Parameter(torch.full((1, n_hidden, 1, 1), 3.0))

    def forward(self, x):
        return x * torch.tanh(self.PSGU_alpha * torch.sigmoid(x))


class AOAF(nn.Module):
    def __init__(self, n_hidden):
        super(AOAF, self).__init__()
        self.n_hidden = n_hidden

    def forward(self, x):
        mean = torch.mean(x)
        return torch.where(x > 0.17 * mean, x - 0.17 * mean, torch.tensor(0.0, device=x.device)) + 0.17 * mean


class SinLU(nn.Module):
    def __init__(self, n_hidden):
        super(SinLU, self).__init__()
        self.SinLU_alpha = nn.Parameter(torch.full((1, n_hidden, 1, 1), 1.0))
        self.SinLU_beta = nn.Parameter(torch.full((1, n_hidden, 1, 1), 1.0))

    def forward(self, x):
        return (x + self.SinLU_alpha * torch.sin(self.SinLU_beta * x)) * torch.sigmoid(x)


class tanhLU(nn.Module):
    def __init__(self, n_hidden):
        super(tanhLU, self).__init__()
        self.tanhLU_alpha = nn.Parameter(torch.full((1, n_hidden, 1, 1), 1.0))
        self.tanhLU_beta = nn.Parameter(torch.full((1, n_hidden, 1, 1), 0.0))
        self.tanhLU_gamma = nn.Parameter(torch.full((1, n_hidden, 1, 1), 1.0))

    def forward(self, x):
        return self.tanhLU_alpha * torch.tanh(self.tanhLU_gamma * x) + self.tanhLU_beta * x


class AQuLU(nn.Module):
    def __init__(self, n_hidden):
        super(AQuLU, self).__init__()
        self.AQuLU_alpha = nn.Parameter(torch.full((1, n_hidden, 1, 1), 7 / 30))
        self.AQuLU_beta = nn.Parameter(torch.full((1, n_hidden, 1, 1), math.sqrt(1 / 2)))

    def forward(self, x):
        return torch.where(
            x >= (1 - self.AQuLU_beta) / self.AQuLU_alpha,
            x,
            torch.where(
                (x >= -self.AQuLU_beta / self.AQuLU_alpha) & (x < (1 - self.AQuLU_beta) / self.AQuLU_alpha),
                (x ** 2) * self.AQuLU_alpha + x * self.AQuLU_beta,
                torch.tensor(0.0, device=x.device)
            )
        )


class RoSwish(nn.Module):
    def __init__(self, n_hidden):
        super(RoSwish, self).__init__()
        self.alpha = nn.Parameter(torch.full((1, n_hidden, 1, 1), 0.817))
        self.beta = nn.Parameter(torch.full((1, n_hidden, 1, 1), 3.000))

    def forward(self, x):
        return (x + self.alpha) * torch.sigmoid(x * self.beta) - self.alpha / 2


activation_fn = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'selu': nn.SELU,
    'mish': nn.Mish,
    'prelu': nn.PReLU,
    'erfact': ErfAct,
    'swish': Swish,
    'psgu': PSGU,
    'aoaf': AOAF,
    'sinlu': SinLU,
    'tanhlu': tanhLU,
    'aqulu': AQuLU,
    'roswish': RoSwish,
}

if activation_fn is None:
    raise ValueError("Unsupported nonlinearity: choose 'relu, gelu, silu, selu, mish, prelu, erfact, swish, "
                             "psgu, aoaf, sinlu, tanhlu, aqulu, or roswish' for nonlinearity_name")


class CNN(nn.Module):
    def __init__(self, activation_fn, out_channels):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            GaussianNoise(sigma=0.15),
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1),
            self.select(activation_fn, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            self.select(activation_fn, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            self.select(activation_fn, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, padding=1),
            self.select(activation_fn, out_channels * 2),
            nn.BatchNorm2d(out_channels * 2),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, padding=1),
            self.select(activation_fn, out_channels * 2),
            nn.BatchNorm2d(out_channels * 2),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, padding=1),
            self.select(activation_fn, out_channels * 2),
            nn.BatchNorm2d(out_channels * 2),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, padding=0),
            self.select(activation_fn, out_channels * 2),
            nn.BatchNorm2d(out_channels * 2),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1),
            self.select(activation_fn, out_channels * 2),
            nn.BatchNorm2d(out_channels * 2),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1),
            self.select(activation_fn, out_channels * 2),
            nn.BatchNorm2d(out_channels * 2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(out_channels * 2, 10)
        self.fc_bn = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_bn(self.fc(x))

    def select(self, activation_fn, n_hiddens):
        if activation_fn == ErfAct or Swish or PSGU or AOAF or SinLU or tanhLU or AQuLU or RoSwish:
            return activation_fn(n_hiddens)
        else:
            return activation_fn()


activation_fn = activation_fn[activation_name]
model = CNN(activation_fn, out_channels=96).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

trainx_white = torch.tensor(trainx_white, dtype=torch.float32).to(device)
trainy = torch.tensor(trainy, dtype=torch.long).to(device)
testx_white = torch.tensor(testx_white, dtype=torch.float32).to(device)
testy = torch.tensor(testy, dtype=torch.long).to(device)

train_dataset = TensorDataset(trainx_white, trainy)
test_dataset = TensorDataset(testx_white, testy)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


for epoch in range(200):
    updated_lr = args.learning_rate * min(2. - epoch / 100., 1.)
    updated_betas = (0.9 if epoch < 100 else 0.5, 0.999)

    for param_group in optimizer.param_groups:
        param_group['lr'] = updated_lr
        param_group['betas'] = updated_betas[:2]

    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    train_loss /= len(train_loader)
    train_err = 1 - correct / total

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        test_loss /= len(test_loader)
        test_err = 1 - correct / total

    print(f'Epoch [{epoch + 1}/200], Train Loss: {train_loss:.4f}, Train Err: {train_err:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Err: {test_err:.4f}')
    results_file.write(f'{epoch + 1}, {train_loss:.4f}, {train_err:.4f}, {test_loss:.4f}, {test_err:.4f}\n')
    results_file.flush()

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"{exp_dir}/network.pth")
        print('Saved')

results_file.close()
