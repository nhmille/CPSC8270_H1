# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 13:51:44 2024

@author: Noah 2020
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GradNormSmall(Exception):
    """Custom exception"""
    pass

# Then optimize wrt accumulated error over epoch
def train(model, space, ground_truth, epochs=1000):
    loss_history = []
    grad_norm_hist = []
    samples = len(space)
    # Iterate for a set number of epochs
    for epoch in range(epochs):
        model.train()
        model.optimizer.zero_grad()
        pred = model(space)
        loss = model.loss(pred, ground_truth)/samples
        loss.backward()
        model.optimizer.step()
        loss_history.append(loss.item())
        
        # Calculate grad norm
        cumulative_gradient = 0
        for p in model.parameters():
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy()**2).sum()
                cumulative_gradient += grad
        grad_norm = cumulative_gradient**0.5
        grad_norm_hist.append(grad_norm)
        
        if grad_norm < 1e-4:
            raise GradNormSmall(f"Gradient norm is small: {grad_norm:.6f}")
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    return loss_history, grad_norm_hist

class NN(nn.Module):
    def __init__(self, learning_rate):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
        # Define optimizer and device
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        # How many parameters does this have? 

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.out(x)
        return x

# Function to check the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def target_function(x):
    return np.pi*x*np.cos(3*x)    












