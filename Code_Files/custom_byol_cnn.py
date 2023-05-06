#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:32:25 2023

@author: ananyakapoor
"""

import torch
from byol_pytorch import BYOL
from torchvision import models
import torch
import torch.nn as nn





# Load libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
import warnings
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, ColumnDataSource
import pickle

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
    
    
    
    
## Preloading Results (If Applicable) 
    
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


files = os.listdir(directory)
all_songs_data = [element for element in files if '.npz' in element] # Get the file paths of each numpy file from Yarden's data
all_songs_data.sort()
os.chdir(directory)

all_audio_data = os.listdir(audio_files)
all_audio_data.sort()


# We will now batch our mini-spectrograms as input for the Resnet50

batch_size = 157
num_batches = 4
window_length = 100
frequency_length = int(15100/window_length)

stacked_windows_torch  = torch.tensor(stacked_windows).float()
stacked_windows_torch = stacked_windows_torch.reshape(num_batches, batch_size, window_length, frequency_length)
x = stacked_windows_torch[0,:, :, :]
x = x.reshape(x.size(0), 1, x.size(1), x.size(2))

resnet = models.resnet50(pretrained=True) # I can modify this to have my own architecture

# Define a simple CNN with one convolutional layer and one fully connected layer
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 14 * 14, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = SimpleCNN()

# import torchvision.transforms as transforms

# data_transform = transforms.Compose([transforms.Resize(100)])

# transformed_images = data_transform(x)


learner = BYOL(
    model,
    image_size = (100,151),
    moving_average_decay = 0.99      # the moving average decay factor for the target encoder, already set at what paper recommends
)

# imgs = torch.randn(10, 3, 256, 256)
projection, embedding = learner(x, return_embedding = True)


