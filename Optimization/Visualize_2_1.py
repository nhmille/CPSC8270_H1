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
from sklearn.decomposition import PCA

from Visualize_2_1_Supp import count_parameters
from Visualize_2_1_Supp import train
from Visualize_2_1_Supp import DNN

# CPSC 8430 HW 1 #2
# Part a: Visualize the optimization process

# Train DNN on MNIST -> Handwritten number ID
# Collect weights of the model every 3 epochs and train 8 times
# Reduce the dimension of weights to 2 by PCA
# Also collect weights of the model of different training events
# Record accuracy (loss) corresponding to collected parameters
# Plot the above results on a figure

# Define device to be GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(str(device))

# Global history to store results of all training runs
weights_8runs = []
layer_8runs = []
acc_8runs = []

# Load MNIST dataset Source:(https://discuss.pytorch.org/t/loading-mnist-from-pytorch/137456/4)
from torchvision import datasets, transforms
# Convert images to tensors and scale values
# Images are 28x28
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)

# Create instance of test_loader for testing dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)

# Train and evaluate models
num_train = 8
for i in range(num_train):
    DeepNN = DNN(learning_rate=0.001)
    # # Print number of parameters
    # if i == 0:
    print(f"Model parameters: {count_parameters(DeepNN)}")
    loss_history, train_acc_history, test_acc_history, weight_history, layer_weight_history = train(DeepNN, train_loader, test_loader, epochs=100)
    
    weight_history_1run = []
    layer_weight_1run = []
    
    weight_history_1run = np.vstack(weight_history)
    layer_weight_1run = np.vstack(layer_weight_history)
    
    pca_full = PCA(n_components=2)
    pca_layer = PCA(n_components=2)

    # Append histories after transforming by PCA
    weights_8runs.append(pca_full.fit_transform(weight_history_1run))
    layer_8runs.append(pca_layer.fit_transform(layer_weight_1run))
    acc_8runs.append(test_acc_history)

weights_8runs = np.array(weights_8runs)
layer_8runs = np.array(layer_8runs)
acc_8runs = np.array(acc_8runs)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(num_train):
    num_points = len(weights_8runs[i])
    plt.scatter(weights_8runs[i][:,0], weights_8runs[i][:,1], 
                color=colors[i], s=1, alpha=0)
    for j in range(num_points):
        plt.text(weights_8runs[i][j,0], weights_8runs[i][j,1], f"{acc_8runs[i][j]:.1f}", fontsize=8, color=colors[i])
plt.title('Whole Model')

plt.subplot(1, 2, 2)
for i in range(num_train):
    num_points = len(layer_8runs[i])
    plt.scatter(layer_8runs[i][:,0], layer_8runs[i][:,1], color=colors[i], s=1, alpha=0)
    for j in range(num_points):
        plt.text(layer_8runs[i][j,0], layer_8runs[i][j,1], 
                 f"{acc_8runs[i][j]:.1f}", fontsize=8, color=colors[i])
plt.title('Hidden Layer #1')
plt.tight_layout()
plt.show()















