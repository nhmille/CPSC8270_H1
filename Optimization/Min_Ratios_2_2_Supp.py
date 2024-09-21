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

def get_hessian(loss, model):
    hessian = []
    for parameter in model.parameters():
        # Compute gradients for each parameter and then combine
        gradients = torch.autograd.grad(loss, parameter, create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(-1)  # Flatten -> Element-wise
        hessian_parameter = []
        
        for gradient in gradients:
            # Compute second-order derivative (Hessian) for the current gradient
            hessian_row = torch.autograd.grad(gradient, parameter, create_graph=True)[0].view(-1)
            hessian_parameter.append(hessian_row)
            
        # Stack all the Hessian rows for this parameter
        hessian.append(torch.stack(hessian_parameter))
    return hessian  # hessian is a list of Hesssians -> one per parameter

def get_grad(model, loss):
    gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    grad_norm = 0
    for gradient in gradients:
        grad_norm += torch.sum(gradient**2)
    grad_norm = torch.sqrt(grad_norm)  # L2 norm of gradients
    return grad_norm

def get_eigs(hessian):
    # Eigenvalues calculation for each parameter's Hessian
    total_positive_eigenvalues = 0
    total_eigenvalues = 0
    for parameter_hessian in hessian:
        eigenvalues, _ = torch.linalg.eigh(parameter_hessian)

        positive_eigenvalues = eigenvalues[eigenvalues > 0]
        total_positive_eigenvalues += len(positive_eigenvalues)
        total_eigenvalues += len(eigenvalues)

    return total_eigenvalues, total_positive_eigenvalues

class NN(nn.Module):
    def __init__(self, learning_rate):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(1, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16,16)
        self.out = nn.Linear(16, 1)
        # Define optimizer and device
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


def train(network, space, ground_truth, epochs=10000):
    low_grad = False
    low_grad_count = 0
    for iteration in range(epochs):
        if iteration%100 == 0:
            print(iteration)
        network.train()
        network.optimizer.zero_grad()
        prediction = network(space)
        loss = network.loss(prediction, ground_truth)/500
        grad_norm = get_grad(network, loss)
        if grad_norm < 1e-4:
            if low_grad == False:
                print("The grad norm is low")
            low_grad = True
        if low_grad:
            low_grad_count += 1
            grad_norm.backward()
            if low_grad_count == 10000:
                break
        else:
            loss.backward()
            network.optimizer.step()

    if low_grad:
        prediction = network(space)
        loss = network.loss(prediction, ground_truth)/500
            
        hessian = get_hessian(loss, network)
        total_eigenvalues, positive_eigenvalues = get_eigs(hessian)

        minimal_ratio = positive_eigenvalues/total_eigenvalues
        print(f"Achieved a low gradient, Minimal Ratio: {minimal_ratio}")

        return loss.item(), minimal_ratio
    return None, None

# Function to check the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def target_function(x):
    return np.pi*x*np.cos(3*x)    












