# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 13:51:44 2024

@author: Noah 2020
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NN_Shallow(nn.Module):
    def __init__(self, learning_rate):
        super(NN_Shallow, self).__init__()
        self.layer1 = nn.Linear(1, 253)   
        self.out = nn.Linear(253, 1)        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.out(x)
        return x

class NN_Deep(nn.Module):
    def __init__(self, learning_rate):
        super(NN_Deep, self).__init__()
        self.layer1 = nn.Linear(1, 19)
        self.layer2 = nn.Linear(19, 18)
        self.layer3 = nn.Linear(18, 18)
        self.out = nn.Linear(18, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.out(x)
        return x

# Then optimize wrt accumulated error over epoch
def train(model, space, ground_truth, epochs=2000):
    loss_history = []
    # Iterate for a set number of epochs
    for epoch in range(epochs):
        # Each epoch, accumulate error over the ground-truth -> Process as batch
        model.train()
        model.optimizer.zero_grad()
        pred = model(space)
        loss_val = model.loss(pred, ground_truth)
        loss_val.backward()
        model.optimizer.step()
        loss_history.append(loss_val.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss_val.item():.2f}")
    return loss_history

# Function to check the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def target_function(x):
    return np.pi*x*np.cos(3*x)    












