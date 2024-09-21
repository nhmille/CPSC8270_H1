# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:09:17 2024

@author: Noah Miller
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from SimulateFunction_Supp_1_1 import count_parameters
from SimulateFunction_Supp_1_1 import target_function
from SimulateFunction_Supp_1_1 import NN_Shallow
from SimulateFunction_Supp_1_1 import NN_Deep
from SimulateFunction_Supp_1_1 import train

# CPSC 8430 HW 1
# Part 1: Deep vs. Shallow
# Part a
# Simulate a single-input single-output nonlinear function using a neural network
# Use a simple trig function like cos(10piX)/(10piX)
 
# Train at least two different DNN with same total number of parameters until convergence
# Plot loss in each epoch 
# Plot ground-truth vs. predictions
# Tips: Constrain input domain, tune hyperparameters carefully

# Define device to be GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(str(device))

# Create array to store ground-truth -> Pass to function
space = np.linspace(0, 2*np.pi, 500).reshape(-1, 1)
ground_truth = target_function(space)

# Convert to tensors
space_tensor = torch.tensor(space, dtype=torch.float32).to(device)
ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32).to(device)

# Initialize deep neural network
shallow = NN_Shallow(learning_rate=0.001)
deep = NN_Deep(learning_rate=0.001)

# Print number of parameters
print(f"Shallow model parameters: {count_parameters(shallow)}")
print(f"Deep model parameters: {count_parameters(deep)}")

# Train models
loss_history_shallow = train(shallow, space_tensor, ground_truth_tensor, epochs=50000)
loss_history_deep = train(deep, space_tensor, ground_truth_tensor, epochs=50000)

# Plot losses during training
plt.figure(dpi=300)
plt.plot(loss_history_shallow, label="Shallow NN Loss")
plt.plot(loss_history_deep, label="Deep NN Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Switch networks into evaluation (inference) mode to get predictions
deep.eval()
shallow.eval()

# Turn off gradient tracking since we are inferring
with torch.no_grad():
    # Collect predictions over batch of training data
    # Move values back to cpu for plotting
    shal_pred = shallow(space_tensor).cpu().numpy()
    deep_pred = deep(space_tensor).cpu().numpy()    
    
# Plot ground truth and predictions
plt.figure(dpi=300)
plt.plot(space, ground_truth, label='Ground Truth')
plt.plot(space, deep_pred, label='Deep NN Predictions')
plt.plot(space, shal_pred, label='Shallow NN Predictions')
plt.legend()
plt.show()
