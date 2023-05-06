#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 19:11:33 2023

@author: ananyakapoor
"""

# Define a different class for each bird. Each class should contain all the 
# data matrices for that particular bird

# TODO: Define a dictionary of colors for each unique syllable. Check if syllable = 20 for bird 1 is the same as syllable = 20 for bird 2

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

class Spectrogram_Reduction:
    def __init__(self, directory):
        self.dir = directory
        files = os.listdir(self.dir)
        self.all_songs_data = [element for element in files if '.npz' in element]
        
    def extract_song_elements(self, number_of_datafiles):
        
        # Initialize the data structures that will hold our info
        stacked_spec = []
        stacked_labels = []
        stacked_times = []
        stacked_frequencies = []
        os.chdir(self.dir)

        for i in np.arange(number_of_datafiles):
            
            dat = np.load(self.all_songs_data[i])
            spec = dat['s']
            times = dat['t']
            frequencies = dat['f']
            labels = dat['labels']
            labels = labels.T

            stacked_spec.append(spec)
            stacked_labels.append(labels)
            stacked_times.append(times)
            stacked_frequencies.append(frequencies)
            
        stacked_spec = np.concatenate(stacked_spec, axis=1)
        stacked_labels = np.concatenate(stacked_labels, axis = 0)
        # stacked_times = np.concatenate(stacked_times, axis = 1)
        # stacked_times = np.sort(stacked_times)
        # 
        # stacked_frequencies = np.concatenate(stacked_frequencies, axis = 0)
        # stacked_frequencies = np.sort(stacked_frequencies)
        # 
        # time_diff = (np.diff(stacked_times[0]))[0,0]
        # frequency_diff = (np.diff(stacked_frequencies[0], axis = 0))[0,0]
        # 
        # 
        # Now get a dictionary of colors for each unique label
        unique_syllables = np.unique(stacked_labels)
        
        # Create a dictionary that maps categories to random colors
        syllable_colors = {category: np.random.rand(3,) for category in unique_syllables}
        
        return stacked_spec, stacked_labels, syllable_colors
    
    # def Plot_Spectrogram(self, spec, times, frequencies, bird_file):
    #     # Plot the spectrogram as a heatmap
        # plt.figure()
        # plt.title(f'Spectrogram for File: {bird_file}',  fontsize=24)
        # plt.pcolormesh(times, frequencies, spec, cmap='jet')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()
    

    def UMAP_Transformation(self, spec,labels, window_size):
        spec_for_UMAP = spec.T
        embedding_arr = []
        window_labels_arr = []
        # for i in range(0, 2):
        for i in range(0, spec_for_UMAP.shape[0]-window_size+1):
            window = spec_for_UMAP[i:i+window_size,:]
            window_labels = labels[i:i+window_size,:]
            
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(window)
            embedding_arr.append(embedding)
            window_labels_arr.append(window_labels)
        
        all_embeddings = np.stack(embedding_arr)
        all_window_labels = np.stack(window_labels_arr )
        
        return all_embeddings, all_window_labels
    
    # def Plot_Projection(self, all_embeddings, all_window_labels, category_colors, reduction_technique):
    #     unique_syllables = np.unique(labels)
        
    #     # Create a scatter plot with randomly color-coded dots
    #     plt.figure()
    #     for category in unique_syllables:
    #         mask = labels == category
    #         mask.shape = (mask.shape[1],)
    #         plt.scatter(reduced_spec[mask,0], reduced_spec[mask,1], c=category_colors[category], label=category)
        
    #     # Add a legend to the plot
    #     plt.legend()
    #     plt.gca().set_aspect('equal', 'datalim')
    #     plt.xlabel("UMAP1")
    #     plt.ylabel("UMAP2")
    #     plt.title('UMAP Projection of the Spectrogram', fontsize=24);
        
    #     plt.show()
        
    def Silhouette_Score(self, reduced_spec, labels):

        score = silhouette_score(reduced_spec, labels)
        
        return score
    
# Set path
directory = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_data_matrices/Python_Files/'   
bird3 = Spectrogram_Reduction(directory = directory)            

UMAP_silhouette_scores_list = []
# file_time_points = []
# unique_syllables = []


stacked_spec, stacked_labels, syllable_colors = bird3.extract_song_elements(2)

# Now let's experiment with the window sizes: we'll stick to the ones Yarden used: 23, 46, 92, 185, 740 

window_size_list = [23]
# window_size_list = [23, 46, 92, 185, 740]
for window_size in window_size_list:
    
    UMAP_embeddings, window_labels = bird3.UMAP_Transformation(stacked_spec, stacked_labels, window_size)
    for k in np.arange(UMAP_embeddings.shape[0]):
        
        reduced_spec = UMAP_embeddings[k,:,:]
        window_labels_reduced = window_labels[k,:,:]
        unique_syllables = np.unique(window_labels_reduced)
        
        SS = bird3.Silhouette_Score(reduced_spec, window_labels_reduced)
        UMAP_silhouette_scores_list.append(SS)


# for k in np.arange(0, 5):
#     plt.figure()
#     for category in unique_syllables:
#         mask = window_labels_reduced == category
#         mask.shape = (mask.shape[0],)
#         plt.scatter(reduced_spec[mask,0], reduced_spec[mask,1], c=syllable_colors[category], label=category)
        
#     plt.legend()
#     plt.gca().set_aspect('equal', 'datalim')
#     plt.xlabel("UMAP1")
#     plt.ylabel("UMAP2")
#     plt.title(f'UMAP Projection, Window # {k}, Silhouette Score: {SS:.2f}', fontsize=24);
#     plt.show()








