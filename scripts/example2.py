import numpy as np
import pandas as pd
from tqdm.auto import trange
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import torch.nn.functional as fn


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = fn.relu(self.layer1(x))
        x = self.drop(x)
        x = fn.relu(self.layer2(x))
        x = self.drop(x)
        x = self.layer3(x)
        return x

    def fit(self, X, y, epochs=500):
        X_torch = Variable(torch.Tensor(X).float())
        y_torch = Variable(torch.Tensor(y).float())
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()
        history = []
        with trange(1, epochs + 1, desc='Training', leave=True) as steps:
            for k in steps:
                y_pred = self.forward(x=X_torch)
                loss = loss_fn(y_pred, y_torch)
                status = {'loss': loss.item()}
                history.append(status['loss'])
                steps.set_postfix(status)
                # Zero gradients
                optimizer.zero_grad()
                loss.backward()  # Gradients
                optimizer.step()  # Update

        return history