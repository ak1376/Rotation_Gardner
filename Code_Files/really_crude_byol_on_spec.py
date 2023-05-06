#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:41:59 2023

@author: ananyakapoor
"""

import torch
from byol_pytorch import BYOL
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim

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

batch_size = 157
num_batches = int(stacked_windows.shape[0]/batch_size)
height = stacked_window_times.shape[1]
width = int(stacked_windows.shape[1]/height)


# x = stacked_windows[0:157,:]
# input_torch = x.reshape(batch_size, 1, height, width)
x = stacked_windows[0:10,:]
input_torch = x.reshape(10, 1, height, width)
input_torch = torch.tensor(input_torch).float()

import torch.nn.functional as F
# Resize the image to a new size
new_size = (256, 256)  # replace with your desired size
resized_tensor = F.interpolate(input_torch, size=new_size, mode='bilinear', align_corners=False)

# Print the shape of the resized tensor
print(resized_tensor.shape)

# extract the first dimension of the tensor
first_dim = resized_tensor.shape[0]

# create a new tensor by repeating the original tensor along the new dimensions
new_tensor = resized_tensor.repeat(1, 3, 1, 1)

# reshape the new tensor to have the desired shape
new_tensor = new_tensor.view(first_dim, 3, 256, 256)


resnet = models.resnet50(pretrained=True)

# Define a simple CNN with one convolutional layer and one fully connected layer
# class MyCNN(nn.Module):
#     def __init__(self):
#         super(MyCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(128 * 32 * 32, 256)
#         self.fc2 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = self.pool(nn.functional.relu(self.conv1(x)))
#         x = self.pool(nn.functional.relu(self.conv2(x)))
#         x = self.pool(nn.functional.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 32 * 32)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# model = MyCNN()

learner = BYOL(
    resnet,
    image_size = 256,
    projection_size = 2,           # the projection size
    projection_hidden_size = 100,   # the hidden dimension of the MLP for both the projection and prediction
    moving_average_decay = 0.99      # the moving average decay factor for the target encoder, already set at what paper recommends
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
num_epoch = 10
for i in range(num_epoch):
    print(i)
    images = new_tensor
    loss = learner(images, return_embedding = False, return_projection = True)
    print(loss)
    if i == num_epoch-1:
        with torch.no_grad():
            projection, embedding = learner(images, return_embedding = True)
            break
    opt.zero_grad()
    loss.backward()
    opt.step()

# Specify an HTML file to save the Bokeh image to.
output_file(filename=f'{folder_name}/Plots/byol.html')

projection_arr = projection.clone().detach().numpy()



# Convert the UMAP embedding to a Pandas Dataframe
spec_df = pd.DataFrame(projection_arr, columns=('x', 'y'))


# Create a ColumnDataSource from the data. This contains the UMAP embedding components and the mean colors per mini-spectrogram
source = ColumnDataSource(data=dict(x = projection_arr[:,0], y = projection_arr[:,1], colors=mean_colors_per_minispec[0:first_dim,:]))


# Create a figure and add a scatter plot
p = figure(width=800, height=600, tools=('pan, box_zoom, hover, reset'))
p.scatter(x='x', y='y', size = 7, color = 'colors', source=source)

hover = p.select(dict(type=HoverTool))
hover.tooltips = """
    <div>
        <h3>@x, @y</h3>
        <div>
            <img
                src="@image" height="100" alt="@image" width="100"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
    </div>
"""

p.add_tools(HoverTool(tooltips="""
"""))


# Set the image path for each data point
source.data['image'] = []
for i in np.arange(spec_df.shape[0]):
    source.data['image'].append(f'{folder_name}/Plots/Window_Plots/Window_{i}.png')

show(p)

save(p)
