#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:32:59 2023

This code will do the following functions: 
    1. Stack spectrograms into a complete dataset, with respect to certain parameters
    2. Find the UMAP representation of the spectrograms
    3. Create an interactive plot of the spectrograms in UMAP space
    4. Create a trajectory over time of the spectrograms in UMAP space

@author: ananyakapoor
"""


# Load libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import silhouette_score
import umap
from sklearn.decomposition import PCA
import warnings
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, ColumnDataSource
import pickle

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
warnings.filterwarnings("ignore")

# Set parameters

directory = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_data_matrices/Python_Files'
analysis_path = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Analysis/UMAP_Analysis'

# Parameters we set
num_spec = 2
window_size = 100
stride = 5

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

stacked_windows = np.load(folder_name+'/stacked_windows.npy')
labels_for_window = np.load(folder_name+'/labels_for_window.npy')
embedding = np.load(folder_name+'/UMAP_Embedding.npy')
masked_frequencies = np.load(analysis_path+'/masked_frequencies_lowthresh_500_highthresh_7000.npy')
stacked_window_times = np.load(folder_name+'/stacked_window_times.npy')
    
# open the file for reading in binary mode
with open(folder_name+'/category_colors.pkl', 'rb') as f:
    # load the dictionary from the file using pickle.load()
    category_colors = pickle.load(f)   
    
mean_colors_per_minispec = np.load(folder_name+'/mean_colors_per_minispec.npy')


# =============================================================================
# # If you're running the analysis for the first time 
# =============================================================================


files = os.listdir(directory)
all_songs_data = [element for element in files if '.npz' in element]
all_songs_data.sort()
os.chdir(directory)


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


    # Let's get rid of higher order frequencies
    mask = (frequencies<7000)&(frequencies>500)
    masked_frequencies = frequencies[mask]

    subsetted_spec = spec[mask.reshape(mask.shape[0],),:]
    
    stacked_labels.append(labels)
    stacked_specs.append(subsetted_spec)

stacked_specs = np.concatenate((stacked_specs), axis = 1)
stacked_labels = np.concatenate((stacked_labels), axis = 0)

# Get a list of unique categories
unique_categories = np.unique(stacked_labels)

# Create a dictionary that maps categories to random colors
category_colors = {category: np.random.rand(3,) for category in unique_categories}

spec_for_analysis = stacked_specs.T
window_labels_arr = []
embedding_arr = []
dx = np.diff(times)[0,0]

stacked_windows = []
stacked_labels_for_window = []
stacked_window_times = []

for i in range(0, spec_for_analysis.shape[0] - window_size + 1, stride):
    window = spec_for_analysis[i:i + window_size, :]
    window_times = dx*np.arange(i, i + window_size)
    window = window.reshape(1, window.shape[0]*window.shape[1])
    labels_for_window = stacked_labels[i:i+window_size, :]
    labels_for_window = labels_for_window.reshape(1, labels_for_window.shape[0]*labels_for_window.shape[1])
    stacked_windows.append(window)
    stacked_labels_for_window.append(labels_for_window)
    stacked_window_times.append(window_times)

stacked_windows = np.stack(stacked_windows, axis = 0)
stacked_windows = np.squeeze(stacked_windows)

stacked_labels_for_window = np.stack(stacked_labels_for_window, axis = 0)
stacked_labels_for_window = np.squeeze(stacked_labels_for_window)

stacked_window_times = np.stack(stacked_window_times, axis = 0)

mean_colors_per_minispec = np.zeros((stacked_labels_for_window.shape[0], 3))
for i in np.arange(stacked_labels_for_window.shape[0]):
    list_of_colors_for_row = [category_colors[x] for x in stacked_labels_for_window[i,:]]
    all_colors_in_minispec = np.array(list_of_colors_for_row)
    mean_color = np.mean(all_colors_in_minispec, axis = 0)
    mean_colors_per_minispec[i,:] = mean_color
    
reducer = umap.UMAP()
embedding = reducer.fit_transform(stacked_windows)

# Let's save all the numpy arrays
np.save(folder_name+'/stacked_windows.npy', stacked_windows)
np.save(folder_name+'/labels_for_window.npy', labels_for_window)
np.save(folder_name+'/UMAP_Embedding.npy', embedding)
np.save(analysis_path+'/masked_frequencies_lowthresh_500_highthresh_7000.npy', masked_frequencies)
np.save(folder_name+'/stacked_window_times.npy', stacked_window_times)
np.save(folder_name+'/mean_colors_per_minispec.npy', mean_colors_per_minispec)

# open a file for writing in binary mode
with open(folder_name+'/category_colors.pkl', 'wb') as f:
    # write the dictionary to the file using pickle.dump()
    pickle.dump(category_colors, f)

# The below function will extract the image from each mini-spectrogram
def embeddable_image(data, window_times, iteration_number):
    

    data.shape = (window_size, int(data.shape[0]/window_size))
    data = data.T 
    window_times = window_times.reshape(1, window_times.shape[0])
    
    plt.pcolormesh(window_times, masked_frequencies, data, cmap='jet')
    # let's save the plt colormesh as an image. We will then use the Image package to open it up again
    plt.savefig(folder_name+'/Plots/Window_Plots/'+f'Window_{iteration_number}.png')

    plt.close()
    
    
for i in np.arange(stacked_windows.shape[0]):
    if i%10 == 0:
        print(f'Iteration {i} of {stacked_windows.shape[0]}')
    data = stacked_windows[i,:]
    window_times = stacked_window_times[i,:]
    embeddable_image(data, window_times, i)
    
# =============================================================================
# # In-Common Code
# =============================================================================


output_file(filename=f'{folder_name}/Plots/umap.html')

spec_df = pd.DataFrame(embedding, columns=('x', 'y'))


# Create a ColumnDataSource from the data
# source = ColumnDataSource(spec_df)
source = ColumnDataSource(data=dict(x = embedding[:,0], y = embedding[:,1], colors=mean_colors_per_minispec))


# Create a figure and add a scatter plot

p = figure(width=800, height=800, tools=('pan, box_zoom, hover, reset'))
p.scatter(x='x', y='y', color = 'colors', source=source)

# p.scatter(x=embedding[:,0], y=embedding[:,1], color = mean_colors_per_minispec)

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


from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

x = embedding[:,0]
y = embedding[:,1]

fig = plt.figure()
plt.xlim(np.min(x)-5, np.max(x)+5)
plt.ylim(np.min(y)-5, np.max(y)+5)


# Generate an array of colors for each point
colors = mean_colors_per_minispec

graph = plt.scatter([], [], marker='o', c=[])

def animate(i):
    if i > x.shape[0]-1:
        i = x.shape[0]-1
        
    # Update the data and colors for the points that have already been plotted
    graph.set_offsets(np.vstack((x[:i+1], y[:i+1])).T)
    graph.set_color(colors[:i+1])
    
    # Plot the new point with its color
    graph.axes.scatter(x[i], y[i], color=colors[i])
    
    return graph,

ani = FuncAnimation(fig, animate, interval=1, blit = True)

plt.show()

# create the writer
writer = FFMpegWriter(fps=30)

# save the animation as a mp4 file
ani.save('animation.mp4', writer=writer)

# ani.save(folder_name+'/trajectory_over_time.gif', writer='pillow')

# from matplotlib.animation import FuncAnimation

# x = embedding[:,0]
# y = embedding[:,1]

# fig = plt.figure()
# plt.xlim(np.min(x)-5, np.max(x)+5)
# plt.ylim(np.min(y)-5, np.max(y)+5)
# graph = plt.scatter([], [], c=[])


# def animate(i):
#     if i > x.shape[0]-1:
#         i = x.shape[0]-1
#     graph.set_offsets(np.vstack((x[:i+1], y[:i+1])).T)

#     # graph.set_data(x[:i+1], y[:i+1])
#     # graph.set_color(mean_colors_per_minispec[i+1,:])
#     graph.set_array(mean_colors_per_minispec)
#     return graph

# ani = FuncAnimation(fig, animate, interval=10)

# plt.show()



from matplotlib.animation import FuncAnimation

x = embedding[:,0]
y = embedding[:,1]

fig = plt.figure()
plt.xlim(np.min(x)-5, np.max(x)+5)
plt.ylim(np.min(y)-5, np.max(y)+5)


# Generate an array of colors for each point
colors = mean_colors_per_minispec

graph = plt.scatter([], [], marker='o', c=[])

def animate(i):
    if i > x.shape[0]-1:
        i = x.shape[0]-1
        
    # Update the data and colors for the points that have already been plotted
    graph.set_offsets(np.vstack((x[:i+1], y[:i+1])).T)
    graph.set_color(colors[:i+1])
    
    # Plot the new point with its color
    graph.axes.scatter(x[i], y[i], color=colors[i])
    
    return graph,

ani = FuncAnimation(fig, animate, interval=1, blit = True)

plt.show()




