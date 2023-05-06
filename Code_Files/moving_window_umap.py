#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 18:20:40 2023

@author: ananyakapoor
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import silhouette_score
import umap
from sklearn.decomposition import PCA
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})


os.chdir('/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_data_matrices/Python_Files')

dat = np.load('llb3_0185_2018_04_24_08_34_45.wav.npz')
array_names = dat.keys()

# for array_name in array_names:
#     print(f"Array name: {array_name}")

spec = dat['s']
times = dat['t']
frequencies = dat['f']
labels = dat['labels']
labels = labels.T

# Plot the spectrogram as a heatmap
plt.figure()
plt.title(f'Spectrogram for bird 3, song: llb3_0185_2018_04_24_08_34_45.wav.npz',  fontsize=24)
plt.pcolormesh(times, frequencies, spec, cmap='jet')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

# Get a list of unique categories
unique_categories = np.unique(labels)

# Create a dictionary that maps categories to random colors
category_colors = {category: np.random.rand(3,) for category in unique_categories}

# Get a list of unique categories
unique_categories = np.unique(labels)

# Create a dictionary that maps categories to random colors
category_colors = {category: np.random.rand(3,) for category in unique_categories}


# =============================================================================
# # Code for moving window
# =============================================================================

# spec_for_analysis = spec.T

# import numpy as np 

# yourArray = np.random.randn(64,64)        # just an example
# winSize = 92

# a = np.zeros((4261, winSize, spec_for_analysis.shape[1])) # a python list to hold the windows
# for i in range(0, spec_for_analysis.shape[0]-winSize+1):
#     window = spec_for_analysis[i:i+winSize,:] # each individual window
#     window_labels = labels[i:i+winSize,:]
    
#     reducer = umap.UMAP()
#     embedding = reducer.fit_transform(window)
    
#     plt.figure()
#     plt.scatter(
#         embedding[:, 0],
#         embedding[:, 1], 
#         c = window_labels.T)
    
#     labs = window_labels.copy()
#     labs.shape = (window_labels.shape[0],)
    
#     score = silhouette_score(embedding, labs)


# =============================================================================
# # Code for stacking spectrograms across
# =============================================================================
    
os.chdir('/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_data_matrices/Python_Files')

dat1 = np.load('llb3_0002_2018_04_23_14_18_03.wav.npz')
# array_names = dat.keys()

# for array_name in array_names:
#     print(f"Array name: {array_name}")

spec = dat1['s']
times = dat1['t']
frequencies = dat1['f']
labels = dat1['labels']
labels = labels.T   

dat2 = np.load('llb3_0003_2018_04_23_14_18_54.wav.npz')   
spec2 = dat2['s']
times2 = dat2['t']
frequencies2 = dat2['f']
labels2 = dat2['labels']
labels2 = labels2.T

stacked_labels = np.vstack((labels, labels2)) 

# Get a list of unique categories
unique_categories = np.unique(stacked_labels)

# Create a dictionary that maps categories to random colors
category_colors = {category: np.random.rand(3,) for category in unique_categories}

c = np.hstack((spec, spec2))

reducer = umap.UMAP()
embedding = reducer.fit_transform(c.T)
embedding.shape


plt.figure()
plt.scatter(
    embedding[:, 0],
    embedding[:, 1], 
    c = stacked_labels.T)


labs = stacked_labels.copy()
score = silhouette_score(embedding, labs)
print(f'UMAP Silhouette Score: {score}')


spec_for_analysis = c.T
winSize = 1000
for i in range(0,1):
# for i in range(0, spec_for_analysis.shape[0]-winSize+1):
    window = spec_for_analysis[i:i+winSize,:] # each individual window
    window_labels = stacked_labels[i:i+winSize,:]
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(window)
    
    plt.figure()
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1], 
        c = window_labels.T)
    
    labs = window_labels.copy()
    labs.shape = (window_labels.shape[0],)
    
    score = silhouette_score(embedding, labs)



# =============================================================================
# # Let's stack 10 spectrograms together
# =============================================================================

files = os.listdir('/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_data_matrices/Python_Files')
all_songs_data = [element for element in files if '.npz' in element]

stacked_spec = []
stacked_labels = []

for i in np.arange(0, 10):
    dat = np.load(all_songs_data[i])
    spec = dat['s']
    times = dat['t']
    frequencies = dat['f']
    labels = dat['labels']
    labels = labels.T
    
    stacked_spec.append(spec)
    stacked_labels.append(labels)

    
stacked_spec = np.concatenate(stacked_spec, axis=1)
stacked_labels = np.concatenate(stacked_labels, axis = 0)












i = 0
dat = np.load(all_songs_data[i])
spec = dat['s']
times = dat['t']
frequencies = dat['f']
labels = dat['labels']
labels = labels.T   

stacked_spec = spec
stacked_labels = labels

for i in np.arange(20,40):
    dat = np.load(all_songs_data[i])
    spec = dat['s']
    times = dat['t']
    frequencies = dat['f']
    labels = dat['labels']
    labels = labels.T
    
    stacked_spec = np.concatenate((stacked_spec, spec), axis = 1)
    stacked_labels = np.concatenate((stacked_labels, labels), axis = 0)

spec_for_analysis = stacked_spec.T
winSize = 5000
for i in range(0,5):
# for i in range(0, spec_for_analysis.shape[0]-winSize+1):
    window = spec_for_analysis[i:i+winSize,:] # each individual window
    window_labels = stacked_labels[i:i+winSize,:]
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(window)
    
    plt.figure()
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1], 
        c = window_labels.T)
    
    labs = window_labels.copy()
    labs.shape = (window_labels.shape[0],)
    
    score = silhouette_score(embedding, labs)













# reducer = umap.UMAP()
# embedding = reducer.fit_transform(stacked_spec.T)

# plt.figure()
# plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1], 
#     c = stacked_labels.T)

# labs = window_labels.copy()
# labs.shape = (window_labels.shape[0],)

# score = silhouette_score(embedding, labs)
    
    
    
    
    

    
    










