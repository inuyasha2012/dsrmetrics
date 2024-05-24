from math import inf

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import trange
from sklearn.model_selection import KFold
import pickle as pkl
from sklearn import preprocessing
from utils import find_skyline_brute_force
from gplearn.genetic import SymbolicRegressor

class MyLinear(nn.Linear):

    def forward(self, input):
        weight = 2 * functional.relu(1 * torch.neg(self.weight)) + self.weight
        return functional.linear(input, weight, self.bias)


class Encoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            MyLinear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            MyLinear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)


class Decoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)


class NN(nn.Module):

    def __init__(self, input_dim, x_dim, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, x_dim, encoder_hidden_dim)
        self.decoder = Decoder(x_dim, input_dim, decoder_hidden_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


if __name__ == '__main__':
    x_feature = 4
    field = 'Education'
    with open('data.pkl', 'rb') as f:
        data = pkl.load(f)
    data = np.array(data[field])
    data = torch.from_numpy(data).float()
    data = preprocessing.MinMaxScaler().fit_transform(data)
    skf = KFold(n_splits=5, shuffle=True, random_state=222).split(data)
    for train_idx, val_idx in skf:
        model = NN(data.size(1), x_feature, 64, 64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        data_loader = DataLoader(dataset=data[train_idx], batch_size=1024, shuffle=True)
        val_data_loader = DataLoader(dataset=data[val_idx], batch_size=1024, shuffle=True)
        logger.info('start')
        min_val_loss = inf
        with trange(500) as t:
            for i in t:
                loss_list = []
                for batch in data_loader:
                    y, z = model(batch)
                    mse_loss = nn.MSELoss(reduction='mean')
                    optimizer.zero_grad()
                    recon_loss = mse_loss(y, batch)
                    loss = recon_loss
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        loss_list.append(recon_loss.item())
                with torch.no_grad():
                    model.eval()
                    train_loss = sum(loss_list) / len(loss_list)
                    y, z = model(data[val_idx])
                    val_loss = (y - data[val_idx]).pow(2).mean().sqrt().item()
                    if min_val_loss > val_loss:
                        min_val_loss = val_loss
                    model.train()
                t.set_postfix({'loss': train_loss, 'val_loss': min_val_loss})

        with torch.no_grad():
            y, z = model(data)
            res = find_skyline_brute_force(z.numpy())
            for idx in range(z.size(1)):
                est_gp = SymbolicRegressor(
                    function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'sin', 'cos', 'tan'),
                    verbose=1
                )
                est_gp.fit(data.numpy(), z.numpy()[:, idx])
                print(est_gp._program)
