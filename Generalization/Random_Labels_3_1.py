# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:09:17 2024

@author: Noah Miller
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from Random_Labels_3_1_Supp import CNN
from Random_Labels_3_1_Supp import train

# CPSC 8430 HW 1 3-1
# Train on an actual task -> Handwritten letter ID - MNIST dataset
 
# Shuffle the labels before training -> Try to fit network with random labels

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

network = CNN(learning_rate=0.0002)

# Load MNIST dataset Source:(https://discuss.pytorch.org/t/loading-mnist-from-pytorch/137456/4)
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Create instance of test_loader for testing dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Shuffle datasets 
targets = train_dataset.targets.numpy()
np.random.shuffle(targets)
train_dataset.targets = torch.tensor(targets)

# Train and evaluate models
train_loss_history, test_loss_history = train(network, train_loader, test_loader, epochs=10000)

plt.figure(figsize=(10,5))
plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss over Epochs')
plt.legend()
plt.show()


















