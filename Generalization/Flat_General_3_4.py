# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:09:17 2024

@author: Noah Miller
"""

import torch
import matplotlib.pyplot as plt

from Flat_General_3_4_Supp import CNN
from Flat_General_3_4_Supp import train

# CPSC 8430 HW1 1-3: Flatness vs. Generalization
# Train five models with different hyperparameters (batch size)
# Record loss and accuracy of all models
# Record sensitivity -> Frobenius norm of gradients of loss to input

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset Source:(https://discuss.pytorch.org/t/loading-mnist-from-pytorch/137456/4)
from torchvision import datasets, transforms
# Convert images to tensors and scale values
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
train_loader3 = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
train_loader4 = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
train_loader5 = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
train_loader6 = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
train_loader7 = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
batch_sizes = [16, 32, 64, 128, 256, 512, 1024]

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

train_losses = []
test_losses = []
train_acc = []
test_acc = []
sensitivities = []

for i in range(7):
    model1 = CNN(learning_rate=0.001)
    print(f"Training {i}")
    if i == 0:
        train_accuracy, test_accuracy, train_loss, test_loss, sensitivity = train(model1, train_loader1, test_loader, batch_sizes[i], epochs=20)
    if i == 1:
        train_accuracy, test_accuracy, train_loss, test_loss, sensitivity = train(model1, train_loader2, test_loader, batch_sizes[i], epochs=20)
    if i == 2:
        train_accuracy, test_accuracy, train_loss, test_loss, sensitivity = train(model1, train_loader3, test_loader, batch_sizes[i], epochs=20)
    if i == 3:
        train_accuracy, test_accuracy, train_loss, test_loss, sensitivity = train(model1, train_loader4, test_loader, batch_sizes[i], epochs=20)
    if i == 4:
        train_accuracy, test_accuracy, train_loss, test_loss, sensitivity = train(model1, train_loader5, test_loader, batch_sizes[i], epochs=20)
    if i == 5:
        train_accuracy, test_accuracy, train_loss, test_loss, sensitivity = train(model1, train_loader6, test_loader, batch_sizes[i], epochs=20)
    if i == 6:
        train_accuracy, test_accuracy, train_loss, test_loss, sensitivity = train(model1, train_loader7, test_loader, batch_sizes[i], epochs=20)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)
    sensitivities.append(sensitivity)

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

# Plot accuracy on the left y axis
ax1.plot(batch_sizes, train_acc, 'b-', label='Train Accuracy')
ax1.plot(batch_sizes, test_acc, 'b--', label='Test Accuracy')
ax1.set_xlabel('Batch Size')
ax1.set_xscale('log')
ax1.set_ylabel('Accuracy (%)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot sensitivity on the right axis
ax2 = ax1.twinx()
ax2.plot(batch_sizes, sensitivities, 'r-', label='Sensitivity')
ax2.set_ylabel('Sensitivity', color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.show()


fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

# Plot loss on the left y axis
ax1.plot(batch_sizes, train_loss, 'b-', label='Train Loss')
ax1.plot(batch_sizes, test_loss, 'b--', label='Test Loss')
ax1.set_xlabel('Batch Size')
ax1.set_xscale('log')
ax1.set_ylabel('Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot sensitivity on the right axis
ax2 = ax1.twinx()
ax2.plot(batch_sizes, sensitivities, 'r-', label='Sensitivity')
ax2.set_ylabel('Sensitivity', color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.show()












