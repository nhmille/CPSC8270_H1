# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:09:17 2024

@author: Noah Miller
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from NumParameters_3_2_Supp import CNN
from NumParameters_3_2_Supp import train
from NumParameters_3_2_Supp import count_parameters


# CPSC 8430 HW 1 3-2
# Train on an actual task -> Handwritten letter ID - MNIST dataset
# Train at least 10 networks with varying numbers of parameters 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parameter_counts = []
train_losses = []
test_losses = []
train_accs = []
test_accs = []

# Load MNIST dataset Source:(https://discuss.pytorch.org/t/loading-mnist-from-pytorch/137456/4)
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)

# Create instance of test_loader for testing dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)

for i in range(12):
    network = CNN(i, learning_rate=0.002)

    # Train and evaluate models
    train_loss, test_loss, train_acc, test_acc = train(network, train_loader, test_loader, epochs=20)

    # Count the number of parameters
    parameter_count = count_parameters(network)
    print(f"The number of parameters in model #{i} is {parameter_count}")

    parameter_counts.append(parameter_count)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

# Plot the parameter counts on the X axis and the losses on the Y
# Do the same in another plot for accuracies

plt.figure(figsize=(12, 5))

# Plot Losses
plt.subplot(1, 2, 1)
plt.scatter(parameter_counts, train_losses, label='train_loss')
plt.scatter(parameter_counts, test_losses, label='test_loss')
plt.xlabel('number of parameters')
plt.ylabel('loss')
plt.legend()
plt.title('model loss')

# Plot Accuracies
plt.subplot(1, 2, 2)
plt.scatter(parameter_counts, train_accs, label='train_acc')
plt.scatter(parameter_counts, test_accs, label='test_acc')
plt.xlabel('number of parameters')
plt.ylabel('accuracy')
plt.legend()
plt.title('model accuracy')

plt.show()
















