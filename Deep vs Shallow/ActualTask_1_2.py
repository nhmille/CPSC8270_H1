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

from ActualTask_1_2_Supp import count_parameters
from ActualTask_1_2_Supp import target_function
from ActualTask_1_2_Supp import CNN_Shallow
from ActualTask_1_2_Supp import CNN_Deep
from ActualTask_1_2_Supp import train

# CPSC 8430 HW 1
# Part 1: Deep vs. Shallow
# Part b
# Train on an actual task -> Handwritten letter ID - MNIST dataset
 
# Train at least two different CNN with same total number of parameters until convergence
# Plot loss and accuracy over the training process
# Plot ground-truth vs. predictions

# Define device to be GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(str(device))

# Initialize deep neural network
shallow = CNN_Shallow(learning_rate=0.0002)
deep = CNN_Deep(learning_rate=0.0002)

# Print number of parameters
print(f"Shallow model parameters: {count_parameters(shallow)}")
print(f"Deep model parameters: {count_parameters(deep)}")

# Load MNIST dataset Source:(https://discuss.pytorch.org/t/loading-mnist-from-pytorch/137456/4)
from torchvision import datasets, transforms
# Convert images to tensors and scale values
# Images are 28x28
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Create instance of test_loader for testing dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Train and evaluate models
loss_history_shallow, acc_train_history_shallow, acc_test_history_shallow = train(shallow, train_loader, test_loader, epochs=50)
loss_history_deep, acc_train_history_deep, acc_test_history_deep = train(deep, train_loader, test_loader, epochs=50)

# Plot losses during training
plt.figure(dpi=300)
plt.plot(loss_history_shallow, label="Shallow NN Loss")
plt.plot(loss_history_deep, label="Deep NN Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.figure(dpi=300)
plt.plot(acc_test_history_shallow, label="CNN1 Testing Accuracy")
plt.plot(acc_train_history_shallow, label="CNN1 Training Accuracy")
plt.plot(acc_test_history_deep, label="CNN2 Testing Accuracy")
plt.plot(acc_train_history_deep, label="CNN2 Training Accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.show()


















