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
    def __init__(self, learning_rate):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3)
        self.hidden1 = nn.Linear(16*22*22, 8)  # Fully connected layer (after conv layers flattening)
        self.out = nn.Linear(8, 10)        # Output layer (10 classes for MNIST)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 16*22*22)
        x = torch.relu(self.hidden1(x))
        x = self.out(x)
        return x

# Function to check the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_loader, test_loader, epochs=5):
    loss_history = []
    acc_train_history = []
    acc_test_history = []    
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
            
            _,predicted = torch.max(outputs, 1)             # Store prediction
            total += labels.size(0)                         # What is this?
            correct += (predicted == labels).sum().item()   # Count total number of correct predictions
        
        avg_epoch_loss = epoch_loss/len(train_loader)
        loss_history.append(avg_epoch_loss)
        
        accuracy_train = 100*correct/total
        acc_train_history.append(accuracy_train)      

        # Evaluate accuracy
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():  # Disable gradient calculation during evaluation
            for images, labels in test_loader:
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)               # Store prediction
                total_test += labels.size(0)                         # Store total tested count
                correct_test += (predictions == labels).sum().item() # Count correct predictions
        
        accuracy_test = 100*correct_test/total_test
        acc_test_history.append(accuracy_test)     
         
        print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy_train}")
    return 



def evaluate(model, train_loader, test_loader):
    for epoch in range(1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(model.device), labels.to(model.device)
            model.optimizer.zero_grad()
            outputs = model(images)
            loss = model.loss(outputs, labels)
            train_loss += loss.item()
            
            _,predicted = torch.max(outputs, 1)   
            total += labels.size(0)    
            correct += (predicted == labels).sum().item()
        
        accuracy_train = 100*correct/total
        
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad(): 
            for images, labels in test_loader:
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = model(images)
                loss = model.loss(outputs, labels)
                test_loss += loss.item()
                
                _, predictions = torch.max(outputs, 1)               # Store prediction
                total_test += labels.size(0)                         # Store total tested count
                correct_test += (predictions == labels).sum().item() # Count correct predictions
        
        accuracy_test = 100*correct_test/total_test
         
    return train_loss/64, accuracy_train, test_loss/64, accuracy_test







