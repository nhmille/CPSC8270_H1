# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 13:51:44 2024

@author: Noah 2020
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CNN(nn.Module):
    def __init__(self, sz, learning_rate):
        super(CNN, self).__init__()
        # Multiple convolutional layers with batch normalization
        self.size = sz+1
        self.conv1 = nn.Conv2d(1, self.size*2, kernel_size=3)
        self.conv2 = nn.Conv2d(self.size*2, self.size*2, kernel_size=3)
        self.conv3 = nn.Conv2d(self.size*2, self.size*2, kernel_size=3)
        self.hidden1 = nn.Linear(self.size*2*22*22, self.size*4)  # Fully connected layer (after conv layers flattening)
        self.out = nn.Linear(self.size*4, 10)        # Output layer (10 classes for MNIST)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, self.size*4*22*22)
        x = torch.relu(self.hidden1(x))
        x = self.out(x)
        return x

# Function to check the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_loader, test_loader, epochs=5):
    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []   
    
    # Each epoch -> Train on whole dataset
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        # Loop to process mini-batches of 64 created by train_loader
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(model.device), labels.to(model.device)
            model.optimizer.zero_grad()
            outputs = model(images)
            loss = model.loss(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            model.optimizer.step()
            
            # Count correct predictions
            _,predicted = torch.max(outputs, 1)          
            total += labels.size(0)                       
            correct += (predicted == labels).sum().item() 
                
        avg_epoch_loss = epoch_loss/len(train_loader)
        train_loss_history.append(avg_epoch_loss)
        accuracy_train = 100*correct/total
        train_acc_history.append(accuracy_train)

        # Evaluate accuracy
        model.eval()
        epoch_test_loss = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():  # Disable gradient calculation during evaluation
            for images, labels in test_loader:
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = model(images)
                loss = model.loss(outputs, labels)
                epoch_test_loss += loss.item()
                
                _, predictions = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predictions == labels).sum().item()
                
        test_loss_history.append(epoch_test_loss/len(train_loader))
        accuracy_test = 100*correct_test/total_test
        test_acc_history.append(accuracy_test)
         
        print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}")
    return train_loss_history[-1], test_loss_history[-1], train_acc_history[-1], test_acc_history[-1]










