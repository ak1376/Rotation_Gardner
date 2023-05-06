#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 20:39:58 2023

@author: ananyakapoor
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn # All the neural network modules
import torch.optim as optim

# Let's define our input and output variable. Let's visualize it by plotting.

x = np.linspace(0,3*np.pi,50)
y = np.sin(x)

plt.figure()
plt.plot(x,y)
plt.xlabel("Time (t)")
plt.ylabel("y")
plt.show()

# We will define a class for our RNN. This will store the model parameters, hyperparameters, and model layers (RNN, feedforward). 

class Simple_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,learning_rate,
                 num_epoch, sequence_length, batch_size,output_size):
        super(Simple_RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.output_size = output_size
        
        self.RNN = nn.RNN(self.input_size, self.hidden_size, self.num_layers,batch_first=True, nonlinearity = 'tanh')
        self.fully_connected=nn.Linear(self.hidden_size, self.output_size, bias = True)
        
    def forward(self, x):
        h0 = torch.randn(self.num_layers,self.batch_size, self.hidden_size)
        output_val, _ = self.RNN(x, h0)
        pred_value = self.fully_connected(output_val)
        return pred_value
    

# Define our model parameters
input_size = 1
hidden_size = 10
num_layers = 1
learning_rate = 0.05
num_epoch = 1000
sequence_length = 10
batch_size = 5
output_size = 1

# Initialize the model
my_RNN_model = Simple_RNN(input_size, hidden_size, num_layers, learning_rate, 
                          num_epoch, sequence_length, batch_size, output_size)

# We want to compute the mean squared loss between our predictions and our actual values
criterion=nn.MSELoss()

# We will use the Adam optimizer. We could use anything else too -- like SGD.
optimizer=optim.Adam(my_RNN_model.parameters(),lr=learning_rate)
y_torch = torch.tensor(y).float()
y_torch = y_torch.reshape(my_RNN_model.batch_size, my_RNN_model.sequence_length, my_RNN_model.input_size)

# We will store the losses for each epoch in a list
loss_list = []

for epoch in np.arange(num_epoch):
    
    # Reshape input into batches
    inputs = torch.tensor(x).float()
    inputs=inputs.reshape(my_RNN_model.batch_size, my_RNN_model.sequence_length, my_RNN_model.input_size)
    
    # Get model predictions
    predictions = my_RNN_model.forward(inputs)
    
    # Calculate loss
    optimizer.zero_grad()
    loss = criterion(y_torch, predictions)
    loss_list.append(loss.item())
    
    # Update our model parameters
    loss.backward()
    optimizer.step()
    
    # Let's plot an animation of our predicted sine curve vs actual sine curve for every 10th epoch
    if epoch%10 ==0: 
        predictions_arr = predictions.detach().numpy().reshape(x.shape[0],1)
        
        plt.clf();
        plt.ion()
        plt.title(f'Computed Function: Epoch {epoch}')
        plt.plot(y,'r-',linewidth=1,label='Target Values')
        plt.plot(predictions_arr,linewidth=1,label='Predictions')
        plt.xlabel('Time (t)')
        plt.ylabel('y')
        plt.legend()
        plt.draw();
        plt.pause(0.05);
        
        print("Epoch Number: "+str(epoch)+", Loss Value: "+str(loss.item())) 


# Let's plot the loss curve across all epochs

plt.figure()
plt.title("Loss Curve")
plt.plot(loss_list)
plt.xlabel("Epoch Number")
plt.ylabel("MSE Loss")
plt.show()
