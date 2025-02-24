import io
import os
import sys
import pickle
import math
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# relu, gelu, silu, selu, mish, prelu, erfact, swish, psgu, aoaf, sinlu, tanhlu, aqulu, or roswish
nonlinearity_name = sys.argv[1] if len(sys.argv) > 1 else 'roswish'
learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3  # 1e-3, 1e-4, 1e-5
p = 0.8


def embeddings_to_dict(filename):
    with io.open(filename, 'r', encoding='utf-8') as f:
        word_vecs = {}
        for line in f:
            line = line.strip('\n').split()
            word_vecs[line[0]] = np.array([float(s) for s in line[1:]])
    return word_vecs

def data_to_mat(filename, vocab, tag_to_number, window_size=1, start_symbol=u'UUUNKKK', one_hot=False, return_labels=True):
    with io.open(filename, 'r', encoding='utf-8') as f:
        x, tweet_words, y = [], [], []
        start = True
        for line in f:
            line = line.strip('\n')
            if len(line) == 0:
                tweet_words.extend([u'</s>'] * window_size)
                tweet_words = [w if w in vocab else u'UUUNKKK' for w in tweet_words]
                for i in range(window_size, len(tweet_words) - window_size):
                    x.append(tweet_words[i-window_size:i+window_size+1])
                tweet_words = []
                start = True
                continue

            word, label = line.split('\t')
            if start:
                tweet_words.extend([start_symbol] * window_size)
                start = False

            tweet_words.append(word)

            if return_labels:
                if one_hot:
                    label_one_hot = len(tag_to_number) * [0]
                    label_one_hot[tag_to_number[label]] += 1
                    y.append(label_one_hot)
                else:
                    y.append(tag_to_number[label])

    return np.array(x), np.array(y)

def word_list_to_embedding(words, embeddings, embedding_dimension=50):
    m, n = words.shape
    words = words.reshape((-1))
    return np.array([embeddings[w] for w in words], dtype=np.float32).reshape(m, n*embedding_dimension)

window_size = 1
tag_to_number = {u'N': 0, u'O': 1, u'S': 2, u'^': 3, u'Z': 4, u'L': 5,
                 u'M': 6, u'V': 7, u'A': 8, u'R': 9, u'!': 10,
                 u'D': 11, u'P': 12, u'&': 13, u'T': 14, u'X': 15,
                 u'Y': 16, u'#': 17, u'@': 18, u'~': 19, u'U': 20,
                 u'E': 21, u'$': 22, u',': 23, u'G': 24}

embeddings = embeddings_to_dict('./data/Tweets/embeddings-twitter.txt')
vocab = embeddings.keys()

xt, yt = data_to_mat('./data/Tweets/tweets-train.txt', vocab, tag_to_number, window_size=window_size, start_symbol=u'</s>')
xdev, ydev = data_to_mat('./data/Tweets/tweets-dev.txt', vocab, tag_to_number, window_size=window_size, start_symbol=u'</s>')
xdtest, ydtest = data_to_mat('./data/Tweets/tweets-devtest.txt', vocab, tag_to_number, window_size=window_size, start_symbol=u'</s>')

data = {
    'x_train': xt, 'y_train': yt,
    'x_dev': xdev, 'y_dev': ydev,
    'x_test': xdtest, 'y_test': ydtest
}


num_epochs = 30
num_tags = 25
hidden_size = 256
batch_size = 16
embedding_dimension = 50
example_size = (2*window_size + 1)*embedding_dimension
num_examples = data['y_train'].shape[0]
num_batches = num_examples // batch_size


class TweetTagger(nn.Module):
    def __init__(self, input_size, hidden_size, num_tags, nonlinearity_name):
        super(TweetTagger, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_tags)

        # ErfAct
        self.ErfAct_alpha = nn.Parameter(torch.full((1, hidden_size), 0.75))
        self.ErfAct_beta = nn.Parameter(torch.full((1, hidden_size), 0.75))

        # Swish
        self.Swish_alpha = nn.Parameter(torch.full((1, hidden_size), 1.0))

        # PSGU
        self.PSGU_alpha = nn.Parameter(torch.full((1, hidden_size), 3.0))

        # SinLU
        self.SinLU_alpha = nn.Parameter(torch.full((1, hidden_size), 1.0))
        self.SinLU_beta = nn.Parameter(torch.full((1, hidden_size), 1.0))

        # tanhLU
        self.tanhLU_alpha = nn.Parameter(torch.full((1, hidden_size), 1.0))
        self.tanhLU_beta = nn.Parameter(torch.full((1, hidden_size), 0.0))
        self.tanhLU_gamma = nn.Parameter(torch.full((1, hidden_size), 1.0))

        # AQuLU
        self.AQuLU_alpha = nn.Parameter(torch.full((1, hidden_size), 7 / 30))
        self.AQuLU_beta = nn.Parameter(torch.full((1, hidden_size), math.sqrt(1 / 2)))

        # RoSwish
        self.alpha = nn.Parameter(torch.full((1, hidden_size), 0.817))
        self.beta = nn.Parameter(torch.full((1, hidden_size), 3.000))

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
        elif nonlinearity_name == 'prelu':
            self.nonlinearity = nn.PReLU()
        elif nonlinearity_name == 'erfact':
            self.nonlinearity = self.ErfAct
        elif nonlinearity_name == 'swish':
            self.nonlinearity = self.Swish
        elif nonlinearity_name == 'psgu':
            self.nonlinearity = self.PSGU
        elif nonlinearity_name == 'aoaf':
            self.nonlinearity = self.AOAF
        elif nonlinearity_name == 'sinlu':
            self.nonlinearity = self.SinLU
        elif nonlinearity_name == 'tanhlu':
            self.nonlinearity = self.tanhLU
        elif nonlinearity_name == 'aqulu':
            self.nonlinearity = self.AQuLU
        elif nonlinearity_name == 'starrelu':
            self.nonlinearity = self.StarReLU
        elif nonlinearity_name == 'roswish':
            self.nonlinearity = self.RoSwish
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

    def forward(self, x, is_training):
        h1 = self.nonlinearity(self.fc1(x))
        # h1 = self.bn1(h1)
        h1 = nn.functional.dropout(h1, p=p, training=is_training)
        h2 = self.nonlinearity(self.fc2(h1))
        # h2 = self.bn2(h2)
        h2 = nn.functional.dropout(h2, p=p, training=is_training)
        return self.fc_out(h2)


model = TweetTagger(example_size, hidden_size, num_tags, nonlinearity_name).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


if not os.path.exists("./data/"):
    os.makedirs("./data/")

if os.path.exists("./data/twitter_pos_" + nonlinearity_name + ".p"):
    history = pickle.load(open("./data/twitter_pos_" + nonlinearity_name + ".p", "rb"))
    key_str = str(len(history)//7 + 1)
    history["lr" + key_str] = learning_rate
    history["train_loss" + key_str] = []
    history["train_err" + key_str] = []
    history["val_loss" + key_str] = []
    history["val_err" + key_str] = []
    history["test_loss" + key_str] = []
    history["test_err" + key_str] = []
else:
    history = {
        "lr1": learning_rate,
        'train_loss1': [], 'train_err1': [], 'val_loss1': [],
        'val_err1': [], 'test_loss1': [], 'test_err1': []
    }
    key_str = '1'


for epoch in range(num_epochs):
    model.train()
    indices = np.arange(num_examples)
    np.random.shuffle(indices)
    data['x_train'] = data['x_train'][indices]
    data['y_train'] = data['y_train'][indices]

    for i in range(num_batches):
        offset = i * batch_size
        bx = word_list_to_embedding(data['x_train'][offset:offset + batch_size, :], embeddings, embedding_dimension)
        by = data['y_train'][offset:offset + batch_size]
        bx_tensor = torch.tensor(bx, dtype=torch.float32)
        by_tensor = torch.tensor(by, dtype=torch.long)
        bx_tensor, by_tensor = bx_tensor.to(device), by_tensor.to(device)

        optimizer.zero_grad()
        logits = model(bx_tensor, is_training=True)
        loss = criterion(logits, by_tensor)
        loss.backward()
        optimizer.step()

        history["train_loss" + key_str].append(loss.item())
        err = (torch.argmax(logits, dim=1) != by_tensor).float().mean().item()
        history["train_err" + key_str].append(err)

        if i % (num_batches // 5) == 0:
            model.eval()
            with torch.no_grad():
                x_dev = torch.tensor(word_list_to_embedding(data['x_dev'], embeddings, embedding_dimension), dtype=torch.float32).to(device)
                y_dev = torch.tensor(data['y_dev'], dtype=torch.long).to(device)
                val_logits = model(x_dev, is_training=False)
                val_loss = (criterion(val_logits, y_dev)).item()
                history["val_loss" + key_str].append(val_loss)
                val_err = (torch.argmax(val_logits, dim=1) != y_dev).float().mean().item()
                history["val_err" + key_str].append(val_err)

                x_test = torch.tensor(word_list_to_embedding(data['x_test'], embeddings, embedding_dimension), dtype=torch.float32).to(device)
                y_test = torch.tensor(data['y_test'], dtype=torch.long).to(device)
                test_logits = model(x_test, is_training=False)
                test_loss = (criterion(test_logits, y_test)).item()
                history["test_loss" + key_str].append(test_loss)
                test_err = (torch.argmax(test_logits, dim=1) != y_test).float().mean().item()
                history["test_err" + key_str].append(test_err)

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Err: {err:.4f}, '
        f'Val Loss: {val_loss:.4f}, Val Err: {val_err:.4f}, '
        f'Test Loss: {test_loss:.4f}, Test Err: {test_err:.4f}')

pickle.dump(history, open("./data/twitter_pos_" + nonlinearity_name + ".p", "wb"))
