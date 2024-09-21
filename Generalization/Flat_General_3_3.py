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

from Flat_General_3_3_Supp import CNN
from Flat_General_3_3_Supp import evaluate
from Flat_General_3_3_Supp import train

# CPSC 8430 HW1 1-3: Flatness vs. Generalization
# Train two model with different hyperparameters (batch size, learning rate)
# Record loss and accuracy of the model which is a linear interpolation between the models
# theta_alpha = (1-alpha)*theta_1 + alpha*theta_2
# alpha is the interpolation ratio and theta is the parameter of the model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model1 = CNN(learning_rate=0.01)
model2 = CNN(learning_rate=0.01)

# Load MNIST dataset Source:(https://discuss.pytorch.org/t/loading-mnist-from-pytorch/137456/4)
from torchvision import datasets, transforms
# Convert images to tensors and scale values
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)

# Create instance of test_loader for testing dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Train and evaluate models
train(model1, train_loader1, test_loader, epochs=25)
train(model2, train_loader2, test_loader, epochs=25)

alphas = np.linspace(-1, 2, num=50)
alpha_train_loss = []
alpha_train_acc = []
alpha_test_loss = []
alpha_test_acc = []

for alpha in alphas:
    print(f"Evaluating alpha {alpha}")
    model_alpha = CNN(learning_rate=0.0002).to(device)
    for theta_1, theta_2, theta_alpha in zip(model1.parameters(), model2.parameters(), model_alpha.parameters()):
        theta_alpha.data = (1-alpha)*theta_1.data + alpha*theta_2.data

    # Evaluate model over test and train dataset
    train_loss, accuracy_train, test_loss, accuracy_test = evaluate(model_alpha, train_loader1, test_loader)

    print(f"Loss {test_loss}, Accuracy: {accuracy_test}")

    alpha_train_loss.append(train_loss)
    alpha_train_acc.append(accuracy_train)
    alpha_test_loss.append(test_loss)
    alpha_test_acc.append(accuracy_test)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(alphas, alpha_train_loss, 'b-', label='Train Loss')
ax1.plot(alphas, alpha_test_loss, 'b--', label='Test Loss')
ax1.set_xlabel('Alpha')
ax1.set_ylabel('Loss', color='blue')
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(alphas, alpha_train_acc, 'r-', label='Train Accuracy')
ax2.plot(alphas, alpha_test_acc, 'r--', label='Test Accuracy')
ax2.set_ylabel('Accuracy (%)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.title('Model Alpha Interpolation')
plt.show()















