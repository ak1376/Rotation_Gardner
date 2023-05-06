#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:59:02 2023

@author: ananyakapoor
"""

# After meeting with Tim, I will work on the following steps:
    # 1. Look at spectrograms but capping it at 5000 Hz . Also would want to filter out the lower basal frequencies
    # 2. Stack a reasonable number of spectrograms together 
    # 3. Identify a window size based on the syllable length 
    # 4. Do a UMAP embedding of the spectrogram. Visualize the embedding
    # 5. If you can, create an animation of the transitions
    

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import silhouette_score
import umap
from sklearn.decomposition import PCA
import warnings
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
warnings.filterwarnings("ignore")

# ===============================================================================================================================
# # Let's play around with filtering out Hz > 5000 and Hz<500. Stacking two Spectrograms, entire spectrogram being used for UMAP
# ===============================================================================================================================

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})


# os.chdir('/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_data_matrices/Python_Files')

# # DATASET 1

# dat1 = np.load('llb3_0185_2018_04_24_08_34_45.wav.npz')
# # array_names = dat.keys()

# spec1 = dat1['s']
# times1 = dat1['t']
# frequencies1 = dat1['f']
# labels1 = dat1['labels']
# labels1 = labels1.T
# # df = pd.DataFrame(labels.T)
# # df.columns = ['labels']

# # Plot the spectrogram as a heatmap
# plt.figure()
# plt.title("Spectrogram",  fontsize=24)
# plt.pcolormesh(times1, frequencies1, spec1, cmap='jet')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()


# # Let's get rid of higher order frequencies
# mask1 = (frequencies1<5000)&(frequencies1>500)
# masked_frequencies1 = frequencies1[mask1]

# subsetted_spec1 = spec1[mask1.reshape(mask1.shape[0],),:]

# plt.figure()
# plt.title("Spectrogram",  fontsize=24)
# plt.pcolormesh(times1, masked_frequencies1, subsetted_spec1, cmap='jet')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# # DATASET 2

# dat2 = np.load('llb3_0186_2018_04_24_08_40_01.wav.npz')
# # array_names = dat.keys()

# spec2 = dat2['s']
# times2 = dat2['t']
# frequencies2 = dat2['f']
# labels2 = dat2['labels']
# labels2 = labels2.T
# # df = pd.DataFrame(labels.T)
# # df.columns = ['labels']

# # Plot the spectrogram as a heatmap
# plt.figure()
# plt.title("Spectrogram",  fontsize=24)
# plt.pcolormesh(times2, frequencies2, spec2, cmap='jet')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()


# # Let's get rid of higher order frequencies
# mask2 = (frequencies2<5000)&(frequencies2>500)
# masked_frequencies2 = frequencies2[mask2]

# subsetted_spec2 = spec2[mask2.reshape(mask2.shape[0],),:]

# plt.figure()
# plt.title("Spectrogram",  fontsize=24)
# plt.pcolormesh(times2, masked_frequencies2, subsetted_spec2, cmap='jet')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# stacked_spectrograms = np.concatenate((subsetted_spec1, subsetted_spec2), axis = 1)

# stacked_labels = np.concatenate((labels1, labels2), axis = 0)

# # Get a list of unique categories
# unique_categories = np.unique(stacked_labels)

# # Create a dictionary that maps categories to random colors
# category_colors = {category: np.random.rand(3,) for category in unique_categories}


# spec_for_analysis = stacked_spectrograms.T

# reducer = umap.UMAP()
# embedding = reducer.fit_transform(spec_for_analysis)
# embedding.shape

# plt.figure()
# for category in unique_categories:
#     mask = stacked_labels == category
#     mask.shape = (mask.shape[0],)
#     plt.scatter(embedding[mask,0], embedding[mask,1], c=category_colors[category], label=category)
    
# plt.legend()
# plt.gca().set_aspect('equal', 'datalim')
# plt.xlabel("UMAP1")
# plt.ylabel("UMAP2")
# plt.title(f'UMAP Projection', fontsize=24);
# plt.show()

# window_labels_arr = []
# window_size = 200
# for i in range(0, 1):
# # for i in range(0, spec_for_analysis.shape[0]-window_size+1):
#     window = spec_for_analysis[i:i+window_size,:]
#     window_labels = stacked_labels[i:i+window_size,:]
    
#     reducer = umap.UMAP()
#     embedding = reducer.fit_transform(window)
#     # embedding_arr.append(embedding)
#     window_labels_arr.append(window_labels)


# unique_categories = np.unique(window_labels)

# plt.figure()
# for category in unique_categories:
#     mask = window_labels == category
#     mask.shape = (mask.shape[0],)
#     plt.scatter(embedding[mask,0], embedding[mask,1], c=category_colors[category], label=category)
    
# plt.legend()
# plt.gca().set_aspect('equal', 'datalim')
# plt.xlabel("UMAP1")
# plt.ylabel("UMAP2")
# plt.title(f'UMAP Projection', fontsize=24);
# plt.show()

# =============================================================================
# # Let's take a step towards automation
# =============================================================================


# NEED TO DO THE FOLLOWING
# 1. DATA DIMENSIONALITY IS [# TIME FREQUENCY BINS, 2D ARRAY RESHAPED TO A VECTOR]

directory = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_data_matrices/Python_Files'

analysis_path = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Analysis'
date_path = 'UMAP_Analysis'
files = os.listdir(directory)
all_songs_data = [element for element in files if '.npz' in element]
all_songs_data.sort()
os.chdir(directory)
num_spec = 1
plt.ioff()

stacked_labels = [] 
stacked_specs = []
# stacked_freq = []
for i in np.arange(num_spec):
    dat = np.load(all_songs_data[i])
    spec = dat['s']
    times = dat['t']
    frequencies = dat['f']
    labels = dat['labels']
    labels = labels.T

    ## Plot the spectrogram as a heatmap
    plt.figure()
    plt.title("Spectrogram",  fontsize=24)
    plt.pcolormesh(times, frequencies, spec, cmap='jet')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    # plt.show()


    # Let's get rid of higher order frequencies
    mask = (frequencies<7000)&(frequencies>500)
    masked_frequencies = frequencies[mask]

    subsetted_spec = spec[mask.reshape(mask.shape[0],),:]
    
    stacked_labels.append(labels)
    stacked_specs.append(subsetted_spec)
    # stacked_freq.append(masked_frequencies)

stacked_specs = np.concatenate((stacked_specs), axis = 1)
stacked_labels = np.concatenate((stacked_labels), axis = 0)
# stacked_freq = np.concatenate((stacked_freq), axis = 0)

# Get a list of unique categories
unique_categories = np.unique(stacked_labels)

# Create a dictionary that maps categories to random colors
category_colors = {category: np.random.rand(3,) for category in unique_categories}

spec_for_analysis = stacked_specs.T
window_labels_arr = []
embedding_arr = []
window_size = 100
dx = np.diff(times)[0,0]
for i in range(0, 50):
    print(i)
# for i in range(0, spec_for_analysis.shape[0]-window_size+1):
    window = spec_for_analysis[i:i+window_size,:]
    window_labels = stacked_labels[i:i+window_size,:]
    # window_freq = stacked_freq[i:i+window_size, :]
    window_times = dx*np.arange(i, i+window_size, 1)
    # Get a list of unique categories
    unique_categories = np.unique(window_labels)

    # Create a dictionary that maps categories to random colors
    category_colors = {category: np.random.rand(3,) for category in unique_categories}
    
    fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols=1, figsize = (14,10))
    
    window_times.shape = (1, window_times.shape[0])
    masked_frequencies.shape = (masked_frequencies.shape[0], 1)
    
    
    ax1.set_title(f'Spectrogram Between {float(window_times[:,i]):.2f} and {float(window_times[:,-1]):.4f} Seconds',  fontsize=24)
    ax1.pcolormesh(window_times, masked_frequencies, window.T, cmap='jet')
    ax2.set_xlim(float(window_times[:,0]), float(window_times[:,-1]))
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_xlabel('Time [sec]')
    
    ax2.set_title("Timecourse of Syllables", fontsize = 24)
    ax2.plot(window_times.T, window_labels)
    ax2.set_xlim(float(window_times[:,0]), float(window_times[:,-1]))
    ax2.set_xlabel("Time [sec]")
    ax2.set_ylabel("Syllable Category (a.u.)")

    plt.savefig((analysis_path+"/"+date_path+"/"+"Plots"+"/"+f'/Window_Size_{window_size}'+"/"+f'Window_{i}_Spec.pdf'))

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(window)
    
    plt.figure()
    for category in unique_categories:
        mask = window_labels == category
        mask.shape = (mask.shape[0],)
        plt.scatter(embedding[mask,0], embedding[mask,1], c=category_colors[category], label=category)
    
    plt.legend()
    # plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f'UMAP Projection Between {float(window_times[:,i]):.4f} and {float(window_times[:,-1]):.4f} Seconds', fontsize=24);
    plt.savefig((analysis_path+"/"+date_path+"/"+"Plots"+"/"+"UMAP_Projections/"+f'Window_{i}_UMAP.pdf'))
    
    # fig.show(savefig=True, fname = (analysis_path+"/"+date_path+"/"+"Plots"+"/"+f'/Window_Size_{window_size}'+"/"+f'Window_{i}.pdf'))


    # plt.figure()
    # oldLabel = np.zeros((1,3))
    # for i in np.arange(window_times.shape[1]):
    #     print(i)
    #     # lab = int(window_labels[i,:])
    #     # color_val = category_colors[lab]
    #     # if oldLabel !=color_val
    #     plt.scatter(embedding[i,0], embedding[i,1], color = category_colors[int(window_labels[i,:])])
    #     plt.pause(0.00000000000001)
    # plt.show()
    
    embedding_arr.append(embedding)
    window_labels_arr.append(window_labels)

all_embeddings = np.stack(embedding_arr)
all_window_labels = np.stack(window_labels_arr )

np.save(analysis_path+'/UMAP_Embeddings_all_windows.npy', all_embeddings)
np.save(analysis_path+'/all_window_labels.npy', all_window_labels)

# k = 0 
# UMAP_embedding = all_embeddings[k,:,:]

# plt.figure()
# for category in unique_categories:
#     mask = window_labels == category
#     mask.shape = (mask.shape[0],)
#     plt.scatter(embedding[mask,0], embedding[mask,1], c=category_colors[category], label=category)

# plt.legend()
# # plt.gca().set_aspect('equal', 'datalim')
# plt.xlabel("UMAP1")
# plt.ylabel("UMAP2")
# plt.title(f'UMAP Projection Between {float(window_times[:,i]):.4f} and {float(window_times[:,-1]):.4f} Seconds', fontsize=24);

# for i in np.arange(UMAP_embedding.shape[0]):
#     plt.scatter(UMAP_embedding[i,0], UMAP_embedding[i,1], color = category_colors[int(window_labels[i,:])])
#     # plt.pause(0.00000000000001)
# plt.show()





# class Spectrogram_Reduction:
#     def __init__(self, directory):
#         self.dir = directory
#         files = os.listdir(self.dir)
#         self.all_songs_data = [element for element in files if '.npz' in element]
        
#     def extract_song_elements(self, number_of_datafiles):
        
#         # Initialize the data structures that will hold our info
#         stacked_spec = []
#         stacked_labels = []
#         stacked_times = []
#         stacked_frequencies = []
#         os.chdir(self.dir)

#         for i in np.arange(number_of_datafiles):
            
#             dat = np.load(self.all_songs_data[i])
#             spec = dat['s']
#             times = dat['t']
#             frequencies = dat['f']
#             labels = dat['labels']
#             labels = labels.T

#             stacked_spec.append(spec)
#             stacked_labels.append(labels)
#             stacked_times.append(times)
#             stacked_frequencies.append(frequencies)
            
#         stacked_spec = np.concatenate(stacked_spec, axis=1)
#         stacked_labels = np.concatenate(stacked_labels, axis = 0)
#         # stacked_times = np.concatenate(stacked_times, axis = 1)
#         # stacked_times = np.sort(stacked_times)
#         # 
#         # stacked_frequencies = np.concatenate(stacked_frequencies, axis = 0)
#         # stacked_frequencies = np.sort(stacked_frequencies)
#         # 
#         # time_diff = (np.diff(stacked_times[0]))[0,0]
#         # frequency_diff = (np.diff(stacked_frequencies[0], axis = 0))[0,0]
#         # 
#         # 
#         # Now get a dictionary of colors for each unique label
#         unique_syllables = np.unique(stacked_labels)
        
#         # Create a dictionary that maps categories to random colors
#         syllable_colors = {category: np.random.rand(3,) for category in unique_syllables}
        
#         return stacked_spec, stacked_labels, syllable_colors


# directory = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_data_matrices/Python_Files/'   
# bird3 = Spectrogram_Reduction(directory = directory)    

# stacked_spec, stacked_labels, syllable_colors = bird3.extract_song_elements(2)












