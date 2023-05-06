#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 11:46:35 2023

@author: ananyakapoor
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, ColumnDataSource
import pickle
import torch
import torch.nn as nn

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
warnings.filterwarnings("ignore")

# Set parameters
bird_dir = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/'
audio_files = bird_dir+'llb3_songs'
directory = bird_dir+ 'llb3_data_matrices/Python_Files'
analysis_path = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Analysis/UMAP_Analysis'

# Parameters we set
num_spec = 1
window_size = 100
stride = 10

# Define the folder name
folder_name = f'{analysis_path}/Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}'

# Create the folder if it doesn't already exist
if not os.path.exists(folder_name+"/Plots/Window_Plots"):
    os.makedirs(folder_name+"/Plots/Window_Plots")
    print(f'Folder "{folder_name}" created successfully.')
else:
    print(f'Folder "{folder_name}" already exists.')
    
    
    
# =============================================================================
# # If you are loading the results from a previous analysis, run the following lines of code
# =============================================================================

stacked_windows = np.load(folder_name+'/stacked_windows.npy') # An array of all the mini-spectrograms
labels_for_window = np.load(folder_name+'/labels_for_window.npy') # The syllable labels for each time point in each mini-spectrogram
embedding = np.load(folder_name+'/UMAP_Embedding.npy') # The pre-computed UMAP embedding (2 dimensional)
masked_frequencies = np.load(analysis_path+'/masked_frequencies_lowthresh_500_highthresh_7000.npy') # The frequencies we want to use for analysis. Excluding unnecessarily low and high frequencies
stacked_window_times = np.load(folder_name+'/stacked_window_times.npy') # The onsets and ending of each mini-spectrogram
    
# open the file for reading in binary mode
with open(folder_name+'/category_colors.pkl', 'rb') as f:
    # load the dictionary from the file using pickle.load()
    category_colors = pickle.load(f)   
    
# Each syllable is given a unique color. Each mini-spectrogram will have an average syllable color associated with it. This is the average RGB value across all unique syllables in the mini-spectrogram
mean_colors_per_minispec = np.load(folder_name+'/mean_colors_per_minispec.npy')


# Define a simple CNN with one convolutional layer and one fully connected layer
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16*128*128, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Create a sample tensor with shape (1, 1, 28, 28)
# batch_size = 1
# num_channels = 1
# height = 28
# width = 28

batch_size = 157
num_batches = int(stacked_windows.shape[0]/batch_size)
height = stacked_window_times.shape[1]
width = int(stacked_windows.shape[1]/height)

x = stacked_windows[0:157,:]
input_torch = x.reshape(batch_size, 1, height, width)
input_torch = torch.tensor(input_torch).float()

import torch.nn.functional as F
# Resize the image to a new size
new_size = (256, 256)  # replace with your desired size
resized_tensor = F.interpolate(input_torch, size=new_size, mode='bilinear', align_corners=False)


# Create an instance of the CNN
model = SimpleCNN()

# Perform the forward pass on the sample tensor

# The are 16 channels 
conv_out = model.conv(resized_tensor)
relu_out = model.relu(conv_out)
pool_out = model.pool(relu_out)
flat_out = torch.flatten(pool_out, 1)
fc_out = model.fc(flat_out)

# Print the shapes of the output tensors at each step
print("Input shape:", x.shape)
print("Conv output shape:", conv_out.shape)
print("ReLU output shape:", relu_out.shape)
print("Pooling output shape:", pool_out.shape)
print("Flattened output shape:", flat_out.shape)
print("Fully connected output shape:", fc_out.shape)


import torch
from byol_pytorch import BYOL
from torchvision import models

learner = BYOL(
    model,
    image_size = 256)



