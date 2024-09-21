# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:09:17 2024

@author: Noah Miller
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from Observe_Grad_Norm_2_2_Supp import count_parameters
from Observe_Grad_Norm_2_2_Supp import target_function
from Observe_Grad_Norm_2_2_Supp import NN
from Observe_Grad_Norm_2_2_Supp import train

# CPSC 8430 HW 1
# Part 2
# Part b: Observe gradient norm during training
# Plot a figure with two subplots over the training, one for gradient norm and the other for loss 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Generate ground-truth data
space = np.linspace(0, 2*np.pi, 500).reshape(-1, 1)
ground_truth = target_function(space)

# Convert ground-truth arrays to tensors, send to GPU
space_tensor = torch.tensor(space, dtype=torch.float32).to(device)
ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32).to(device)

# Initialize neural network
network = NN(learning_rate=0.005)

# Print number of parameters
print(f"Model parameters: {count_parameters(network)}")

# Train models
loss_history_deep, grad_norm_hist = train(network, space_tensor, ground_truth_tensor, epochs=20000)
    
# Plot loss and gradient norm on one figure
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 2)
plt.plot(loss_history_deep, label='Loss')
plt.title('Loss During')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(2, 1, 1)
plt.plot(grad_norm_hist, label='Gradient Norm', color='r')
plt.title('Gradient Norm')
plt.xlabel('Epochs')
plt.ylabel('Gradient Norm')
plt.tight_layout()
plt.show()

























