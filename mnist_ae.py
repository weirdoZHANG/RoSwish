import os
import sys
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# relu, gelu, silu, selu, mish, prelu, erfact, swish, psgu, aoaf, sinlu, tanhlu, aqulu, or roswish
nonlinearity_name = sys.argv[1] if len(sys.argv) > 1 else 'roswish'
learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3  # 1e-3, 1e-4, 1e-5

training_epochs = 70
batch_size = 64
image_pixels = 28 * 28
n_hiddens = [1000, 500, 250, 30]


class ErfAct(nn.Module):
    def __init__(self, n_hidden):
        super(ErfAct, self).__init__()
        self.ErfAct_alpha = nn.Parameter(torch.full((1, n_hidden), 0.75))
        self.ErfAct_beta = nn.Parameter(torch.full((1, n_hidden), 0.75))

    def forward(self, x):
        return x * torch.erf(self.ErfAct_alpha * torch.exp(self.ErfAct_beta * x))


class Swish(nn.Module):
    def __init__(self, n_hidden):
        super(Swish, self).__init__()
        self.Swish_alpha = nn.Parameter(torch.full((1, n_hidden), 1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.Swish_alpha * x)


class PSGU(nn.Module):
    def __init__(self, n_hidden):
        super(PSGU, self).__init__()
        self.PSGU_alpha = nn.Parameter(torch.full((1, n_hidden), 3.0))

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
        self.SinLU_alpha = nn.Parameter(torch.full((1, n_hidden), 1.0))
        self.SinLU_beta = nn.Parameter(torch.full((1, n_hidden), 1.0))

    def forward(self, x):
        return (x + self.SinLU_alpha * torch.sin(self.SinLU_beta * x)) * torch.sigmoid(x)


class tanhLU(nn.Module):
    def __init__(self, n_hidden):
        super(tanhLU, self).__init__()
        self.tanhLU_alpha = nn.Parameter(torch.full((1, n_hidden), 1.0))
        self.tanhLU_beta = nn.Parameter(torch.full((1, n_hidden), 0.0))
        self.tanhLU_gamma = nn.Parameter(torch.full((1, n_hidden), 1.0))

    def forward(self, x):
        return self.tanhLU_alpha * torch.tanh(self.tanhLU_gamma * x) + self.tanhLU_beta * x


class AQuLU(nn.Module):
    def __init__(self, n_hidden):
        super(AQuLU, self).__init__()
        self.AQuLU_alpha = nn.Parameter(torch.full((1, n_hidden), 7 / 30))
        self.AQuLU_beta = nn.Parameter(torch.full((1, n_hidden), math.sqrt(1 / 2)))

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
        self.alpha = nn.Parameter(torch.full((1, n_hidden), 0.817))
        self.beta = nn.Parameter(torch.full((1, n_hidden), 3.000))

    def forward(self, x):
        return (x + self.alpha) * torch.sigmoid(x * self.beta) - self.alpha / 2


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_pixels, n_hiddens[0]),
            self.select(n_hiddens[0]),
            nn.Linear(n_hiddens[0], n_hiddens[1]),
            self.select(n_hiddens[1]),
            nn.Linear(n_hiddens[1], n_hiddens[2]),
            self.select(n_hiddens[2]),
            nn.Linear(n_hiddens[2], n_hiddens[3]),
            self.select(n_hiddens[3]),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_hiddens[3], n_hiddens[2]),
            self.select(n_hiddens[2]),
            nn.Linear(n_hiddens[2], n_hiddens[1]),
            self.select(n_hiddens[1]),
            nn.Linear(n_hiddens[1], n_hiddens[0]),
            self.select(n_hiddens[0]),
            nn.Linear(n_hiddens[0], image_pixels),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def select(self, n_hiddens):
        if nonlinearity == ErfAct or Swish or PSGU or AOAF or SinLU or tanhLU or AQuLU or RoSwish:
            return nonlinearity(n_hiddens)
        else:
            return nonlinearity()


nonlinearities = {
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

if nonlinearity_name not in nonlinearities:
    raise ValueError("Unsupported nonlinearity: choose 'relu, gelu, silu, selu, mish, prelu, erfact, swish, "
                             "psgu, aoaf, sinlu, tanhlu, aqulu, or roswish' for nonlinearity_name")

transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False)

nonlinearity = nonlinearities[nonlinearity_name]
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


if os.path.exists("./data/mnist_ae_" + nonlinearity_name + ".p"):
    history = pickle.load(open("./data/mnist_ae_" + nonlinearity_name + ".p", "rb"))
    key_str = str(len(history)//3 + 1)
    history["lr" + key_str] = learning_rate
    history["train_loss" + key_str] = []
    history["test_loss" + key_str] = []
else:
    history = {
        "lr1": learning_rate, 'train_loss1': [], 'test_loss1': [],
    }
    key_str = '1'


for epoch in range(training_epochs):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.view(-1, image_pixels)
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    history["train_loss" + key_str].append(train_loss)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            inputs = inputs.view(-1, image_pixels)
            inputs = inputs.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    history["test_loss" + key_str].append(test_loss)

    print(f'Epoch [{epoch + 1}/{training_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

pickle.dump(history, open("./data/mnist_ae_" + nonlinearity_name + ".p", "wb"))
