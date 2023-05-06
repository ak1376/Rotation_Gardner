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
# df = pd.DataFrame(labels.T)
# df.columns = ['labels']

# Plot the spectrogram as a heatmap
plt.figure()
plt.title("Spectrogram",  fontsize=24)
plt.pcolormesh(times, frequencies, spec, cmap='jet')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# Let's do a PCA

spec_for_PCA = spec.T

# from sklearn.preprocessing import StandardScaler
# spec_for_PCA = StandardScaler().fit_transform(spec_for_PCA)

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

pca = PCA(n_components=2)
pca.fit(spec_for_PCA)
X_pca = pca.transform(spec_for_PCA)

# Get a list of unique categories
unique_categories = np.unique(labels)

# Create a dictionary that maps categories to random colors
category_colors = {category: np.random.rand(3,) for category in unique_categories}


# plt.figure()
# plt.scatter(
#     X_pca[:, 0],
#     X_pca[:, 1], 
#     c=category_colors)
# # # plt.colorbar()


# Get a list of unique categories
unique_categories = np.unique(labels)

# Create a dictionary that maps categories to random colors
category_colors = {category: np.random.rand(3,) for category in unique_categories}

# Create a scatter plot with randomly color-coded dots
for category in unique_categories:
    mask = labels == category
    mask.shape = (mask.shape[0],)
    plt.scatter(X_pca[mask,0], X_pca[mask,1], c=category_colors[category], label=category)

# Add a legend to the plot
plt.legend()
plt.gca().set_aspect('equal', 'datalim')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title('PCA projection of the Spectrogram', fontsize=24);

# Show the plot
plt.show()

labs = labels.copy()
labs.shape = (labels.shape[0],)
# Calculate the silhouette score
score = silhouette_score(X_pca, labs)
print(f'PCA Silhouette Score: {score}')

# OCA is pretty bad

# Let's try UMAP

reducer = umap.UMAP()
embedding = reducer.fit_transform(spec_for_PCA)
embedding.shape


plt.figure()
plt.scatter(
    embedding[:, 0],
    embedding[:, 1], 
    c = labels.T)
    # c=[sns.color_palette()[x] for x in df.labels.map({"0":0, "1":1, "3":2, "8":3, "10":4, "18":5})])



plt.gca().set_aspect('equal', 'datalim')
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title('UMAP projection of the Spectrogram', fontsize=24);

labs = labels.copy()
labs.shape = (labels.shape[1],)
# Calculate the silhouette score
score = silhouette_score(embedding, labs)
print(f'UMAP Silhouette Score: {score}')

