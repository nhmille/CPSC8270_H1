# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 13:51:44 2024

@author: Noah 2020
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define shallow neural network
class DNN(nn.Module):
    def __init__(self, learning_rate):
        super(DNN, self).__init__()
        self.hidden1 = nn.Linear(28*28, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 10)      

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)     

    def forward(self, x):
        x = x.view(-1, 28*28)  # MNIST images are 28x28 -> Make into a vector after input
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.out(x)
        return x

# Function to check the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, train_loader, test_loader, epochs=4):
    loss_history = []
    train_acc_history = []
    test_acc_history = []    
    weight_history = []
    layer_weight_history = []
    
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
            
            _,predicted = torch.max(outputs, 1)             # Store prediction
            total += labels.size(0)                         # Increment total predictions made
            correct += (predicted == labels).sum().item()   # Count total number of correct predictions
        
        avg_loss = epoch_loss/len(train_loader)
        loss_history.append(avg_loss)
        
        accuracy_train = 100*correct/total
        train_acc_history.append(accuracy_train)      

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
    
        if (epoch+1)%3 == 0:
            test_acc_history.append(accuracy_test)     
         
        # Collect weights
        if (epoch+1)%3 == 0:
            weights = []
            layer_weights = []
            
            for param in model.parameters():
                weights.append(param.view(-1).cpu().detach().numpy())   
            
            for param in model.hidden1.parameters():
                layer_weights.append(param.view(-1).cpu().detach().numpy())
            
            layer_weight_history.append(np.concatenate(layer_weights))          # Concatenate and store weights     
            weight_history.append(np.concatenate(weights))          # Concatenate and store weights     

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Accuracy: {accuracy_train:.2f}%, Test Accuracy: {accuracy_test:.2f}%")
    return loss_history, train_acc_history, test_acc_history, weight_history, layer_weight_history










