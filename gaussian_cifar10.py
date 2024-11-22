import argparse
import pickle
import sys
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from numpy.linalg import svd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

activation_name = sys.argv[1] if len(sys.argv) > 1 else 'roswish'  # 'relu', 'gelu', 'silu', 'selu', 'mish' or 'roswish'

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
    'roswish': RoSwish,
}


if activation_fn is None:
    raise ValueError(
        "Unsupported nonlinearity: choose 'relu', 'gelu', 'silu', 'selu', 'mish' or 'roswish' for nonlinearity_name")


class CNN(nn.Module):
    def __init__(self, activation_fn, out_channels):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            GaussianNoise(sigma=0.15),
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1),
            RoSwish(out_channels) if activation_fn == RoSwish else activation_fn(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            RoSwish(out_channels) if activation_fn == RoSwish else activation_fn(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            RoSwish(out_channels) if activation_fn == RoSwish else activation_fn(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, padding=1),
            RoSwish(out_channels * 2) if activation_fn == RoSwish else activation_fn(),
            nn.BatchNorm2d(out_channels * 2),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, padding=1),
            RoSwish(out_channels * 2) if activation_fn == RoSwish else activation_fn(),
            nn.BatchNorm2d(out_channels * 2),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, padding=1),
            RoSwish(out_channels * 2) if activation_fn == RoSwish else activation_fn(),
            nn.BatchNorm2d(out_channels * 2),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, padding=0),
            RoSwish(out_channels * 2) if activation_fn == RoSwish else activation_fn(),
            nn.BatchNorm2d(out_channels * 2),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1),
            RoSwish(out_channels * 2) if activation_fn == RoSwish else activation_fn(),
            nn.BatchNorm2d(out_channels * 2),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1),
            RoSwish(out_channels * 2) if activation_fn == RoSwish else activation_fn(),
            nn.BatchNorm2d(out_channels * 2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(out_channels * 2, 10)
        self.fc_bn = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_bn(self.fc(x))


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
