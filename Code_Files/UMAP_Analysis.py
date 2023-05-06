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
        
    def extract_song_elements(self, filename):
        os.chdir(self.dir)
        dat = np.load(filename)
        spec = dat['s']
        times = dat['t']
        frequencies = dat['f']
        labels = dat['labels']
        
        # Now get a dictionary of colors for each unique label
        unique_syllables = np.unique(labels)
        
        # Create a dictionary that maps categories to random colors
        syllable_colors = {category: np.random.rand(3,) for category in unique_syllables}
        
        return spec, times, frequencies, labels, syllable_colors
    
    def Plot_Spectrogram(self, spec, times, frequencies, bird_file):
        # Plot the spectrogram as a heatmap
        plt.figure()
        plt.title(f'Spectrogram for File: {bird_file}',  fontsize=24)
        plt.pcolormesh(times, frequencies, spec, cmap='jet')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    
    
    def PCA_Transformation(self, spec):
        spec_for_PCA = spec.T
        pca = PCA(n_components=2)
        pca.fit(spec_for_PCA)
        X_pca = pca.transform(spec_for_PCA)
        return X_pca
    
    def UMAP_Transformation(self, spec):
        spec_for_UMAP = spec.T
        reducer = umap.UMAP(n_components=2)
        embedding = reducer.fit_transform(spec_for_UMAP)
        return embedding
    
    def Plot_Projection(self, reduced_spec, labels, category_colors, reduction_technique):
        unique_syllables = np.unique(labels)
        
        # Create a scatter plot with randomly color-coded dots
        plt.figure()
        for category in unique_syllables:
            mask = labels == category
            mask.shape = (mask.shape[1],)
            plt.scatter(reduced_spec[mask,0], reduced_spec[mask,1], c=category_colors[category], label=category)
        
        # Add a legend to the plot
        plt.legend()
        plt.gca().set_aspect('equal', 'datalim')
        if reduction_technique == "PCA":
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title('PCA projection of the Spectrogram', fontsize=24);
        elif reduction_technique == "UMAP":
            plt.xlabel("UMAP1")
            plt.ylabel("UMAP2")
            plt.title('UMAP Projection of the Spectrogram', fontsize=24);
        
        plt.show()
        
    def Silhouette_Score(self, labels, reduced_spec):
        labs = labels.copy()
        labs.shape = (labels.shape[1],)
        
        # Calculate the silhouette score
        score = silhouette_score(reduced_spec, labs)
        
        return score
    
# Set path
directory = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_data_matrices/Python_Files/'   
bird3 = Spectrogram_Reduction(directory = directory)            

PCA_silhouette_scores_list = []
UMAP_silhouette_scores_list = []
file_time_points = []
unique_syllables = []

for f in np.arange(0,5):
# for f in np.arange(len(bird3.all_songs_data)):
    if f%1 == 0: 
        print(f' Number of files processed: {f}')
    
    filename = bird3.all_songs_data[f]  
    bird_file = filename.split('.')[0]
    spec, times, frequencies, labels, syllable_colors = bird3.extract_song_elements(filename)
    file_time_points.append(times)
    # Plot Spectrogram
    # bird3.Plot_Spectrogram(spec, times, frequencies, bird_file)
    
    # PCA Analysis
    X_pca = bird3.PCA_Transformation(spec)
    # bird3.Plot_Projection(X_pca, labels, syllable_colors, "PCA")    
    pca_silhouette_score = bird3.Silhouette_Score(labels, X_pca) 
    
    PCA_silhouette_scores_list.append(pca_silhouette_score)
    
    # UMAP Analysis
    X_UMAP = bird3.UMAP_Transformation(spec)
    # bird3.Plot_Projection(X_UMAP, labels, syllable_colors, "UMAP")    
    UMAP_silhouette_score = bird3.Silhouette_Score(labels, X_UMAP) 
    UMAP_silhouette_scores_list.append(UMAP_silhouette_score)

bird3.pca_silhouette_scores = PCA_silhouette_scores_list
bird3.UMAP_silhouette_scores = UMAP_silhouette_scores_list






