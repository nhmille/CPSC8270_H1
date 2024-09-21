import torch
import numpy as np
import matplotlib.pyplot as plt

from Min_Ratios_2_2_Supp import count_parameters
from Min_Ratios_2_2_Supp import target_function
from Min_Ratios_2_2_Supp import NN
from Min_Ratios_2_2_Supp import train

# CPSC 8430 HW 1
# Part 2
# Part c: What happens when the gradient is almost zero?
# Try to find the weights of the model when the grad norm is zero (as small as possible)
# First train with original loss function -> Change obj to grad norm and keep training

# Compute the minimal ratio of the weights -> How likely the weights are to be a minima

# Train 100 times
# Plot the figure between minimal ratio and the loss when the gradient is almost zero for all 100
# Tips: Train on a small network 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Generate ground-truth data
space = np.linspace(0, 2*np.pi, 500).reshape(-1, 1)
ground_truth = target_function(space)

# Convert ground-truth arrays to tensors, send to GPU
space_tensor = torch.tensor(space, dtype=torch.float32).to(device)
ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32).to(device)

losses = []
min_ratios = []

# Train model 100 times
while len(losses) < 100:
    network = NN(learning_rate=0.005)
    if len(losses) == 0:
        print(f"Model parameters: {count_parameters(network)}")    
    print(f"Training {len(losses)+1}")
    loss, minimal_ratio = train(network, space_tensor, ground_truth_tensor, epochs=100000)
    # If the training was unsuccessful in getting low grad -> don't error out and retry that i
    if loss != None and minimal_ratio != None:
        losses.append(loss)
        min_ratios.append(minimal_ratio)
    
# Plot loss vs minimal ratio
plt.figure(figsize=(8, 6))
plt.scatter(min_ratios, losses, color='b')
plt.xlabel('Minimal Ratio')
plt.ylabel('Loss')
plt.title('Loss vs Minimal Ratio')
plt.grid(True)
plt.show()

