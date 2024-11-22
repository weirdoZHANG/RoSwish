import sys
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nonlinearity_name = sys.argv[1] if len(sys.argv) > 1 else 'roswish'  # 'relu', 'gelu', 'silu', 'selu', 'mish' or 'roswish'
learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3

training_epochs = 50
batch_size = 128

n_hidden = 128
n_labels = 10
image_pixels = 28 * 28


class FullyConnectedNetwork(nn.Module):
    def __init__(self, n_hidden, n_labels, nonlinearity_name):
        super(FullyConnectedNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(image_pixels, n_hidden))
        for _ in range(7):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.layers.append(nn.Linear(n_hidden, n_labels))

        self.a = nn.Parameter(torch.full((1, n_hidden), 1.0))
        self.b = nn.Parameter(torch.full((1, n_hidden), 0.0))

        self.alpha = nn.Parameter(torch.full((1, n_hidden), 0.817))
        self.beta = nn.Parameter(torch.full((1, n_hidden), 3.000))

        if nonlinearity_name == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity_name == 'gelu':
            self.nonlinearity = nn.GELU()
        elif nonlinearity_name == 'silu':
            self.nonlinearity = nn.SiLU()
        elif nonlinearity_name == 'selu':
            self.nonlinearity = nn.SELU()
        elif nonlinearity_name == 'mish':
            self.nonlinearity = nn.Mish()
        elif nonlinearity_name == 'roswish':
            self.nonlinearity = self.RoSwish
        else:
            raise ValueError(
                "Unsupported nonlinearity: choose 'relu', 'gelu', 'silu', 'selu', 'mish' or 'roswish' for nonlinearity_name")

    def RoSwish(self, x):
        return (x + self.alpha) * torch.sigmoid(x * self.beta) - self.alpha/2

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            f = self.a * x + self.b * torch.sin(x)
            x = self.nonlinearity(f)
        return self.layers[-1](x)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = FullyConnectedNetwork(n_hidden, n_labels, nonlinearity_name).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


if not os.path.exists("./data/"):
    os.makedirs("./data/")

if os.path.exists("./data/mnist_fcn_NsSin" + nonlinearity_name + ".p"):
    history = pickle.load(open("./data/mnist_fcn_NsSin" + nonlinearity_name + ".p", "rb"))
    key_str = str(len(history)//5 + 1)
    history["lr" + key_str] = learning_rate
    history["train_loss" + key_str] = []
    history["train_err" + key_str] = []
    history["test_loss" + key_str] = []
    history["test_err" + key_str] = []
else:
    history = {
        "lr1": learning_rate,
        'train_loss1': [], 'train_err1': [],
        'test_loss1': [], 'test_err1': []
    }
    key_str = '1'


for epoch in range(training_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        bx = bx.view(-1, image_pixels)
        optimizer.zero_grad()
        outputs = model(bx)
        loss = criterion(outputs, by)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += by.size(0)
        correct += (predicted == by).sum().item()

    train_loss /= len(train_loader)
    train_err = 1 - correct / total
    history["train_loss" + key_str].append(train_loss)
    history["train_err" + key_str].append(train_err)

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            bx = bx.view(-1, image_pixels)
            outputs = model(bx)
            loss = criterion(outputs, by)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += by.size(0)
            correct += (predicted == by).sum().item()

    test_loss /= len(test_loader)
    test_err = 1 - correct / total
    history["test_loss" + key_str].append(test_loss)
    history["test_err" + key_str].append(test_err)

    print(
        f'Epoch [{epoch + 1}/{training_epochs}], Train Loss: {train_loss:.4f}, Train Err: {train_err:.4f},'
        f' Test Loss: {test_loss:.4f}, Test Err: {test_err:.4f}')

pickle.dump(history, open("./data/mnist_fcn_NsSin" + nonlinearity_name + ".p", "wb"))
