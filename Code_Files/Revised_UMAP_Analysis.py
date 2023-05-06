#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:47:43 2023

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
import warnings
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
warnings.filterwarnings("ignore")

# ===============================================================================================================================
# # Let's play around with filtering out Hz > 5000 and Hz<500. Stacking two Spectrograms, entire spectrogram being used for UMAP
# ===============================================================================================================================

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

directory = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_data_matrices/Python_Files'

analysis_path = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Analysis/UMAP_Analysis'

files = os.listdir(directory)
all_songs_data = [element for element in files if '.npz' in element]
all_songs_data.sort()
os.chdir(directory)
num_spec = 10


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

    # Plot the spectrogram as a heatmap
    # plt.figure()
    # plt.title("Spectrogram",  fontsize=24)
    # plt.pcolormesh(times, frequencies, spec, cmap='jet')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
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

stride = 5

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

# Define the folder name
folder_name = f'{analysis_path}/Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}'

# Create the folder if it doesn't already exist
if not os.path.exists(folder_name+"/Plots/Window_Plots"):
    os.makedirs(folder_name+"/Plots/Window_Plots")
    print(f'Folder "{folder_name}" created successfully.')
else:
    print(f'Folder "{folder_name}" already exists.')


reducer = umap.UMAP()
embedding = reducer.fit_transform(stacked_windows)

# Let's save all the numpy arrays
np.save(folder_name+'/stacked_windows.npy', stacked_windows)
np.save(folder_name+'/stacked_labels.npy', stacked_labels)
np.save(folder_name+'/UMAP_Embedding.npy', embedding)


# It's plotting time 

# def embeddable_image(data, window_times):
    
#     # data = (data * 255).astype(np.uint8)
#     # data.shape = (window_size, int(data.shape[0]/window_size))
#     # data = data.T
#     # # convert to uint8
#     # data = np.uint8(data)
#     data.shape = (window_size, int(data.shape[0]/window_size))
#     data = data.T 
#     window_times = window_times.reshape(1, window_times.shape[0])
    
#     plt.pcolormesh(window_times, masked_frequencies, data, cmap='jet')
#     # let's save the plt colormesh as an image. We will then use the Image package to open it up again
#     plt.savefig(analysis_path+"/"+date_path+"/"+"Plots/Window_Spectrograms/"+f'Window.png')

#     image = Image.open(analysis_path+"/"+date_path+"/"+"Plots/Window_Spectrograms/"+f'Window.png')
#     # image = Image.fromarray(data)
#     # image = image.convert('RGB')
#     # show PIL image
#     im_file = BytesIO()
#     img_save = image.save(im_file, format='PNG')
#     # img_save = image.save(analysis_path+"/"+date_path+"/"+"Plots/"+im_file, format='PNG')
#     im_bytes = im_file.getvalue()

#     img_str = "data:image/png;base64," + base64.b64encode(im_bytes).decode()
#     plt.close()
#     return img_str

def embeddable_image(data, window_times, iteration_number):
    
    # data = (data * 255).astype(np.uint8)
    # data.shape = (window_size, int(data.shape[0]/window_size))
    # data = data.T
    # # convert to uint8
    # data = np.uint8(data)
    data.shape = (window_size, int(data.shape[0]/window_size))
    data = data.T 
    window_times = window_times.reshape(1, window_times.shape[0])
    
    plt.pcolormesh(window_times, masked_frequencies, data, cmap='jet')
    # let's save the plt colormesh as an image. We will then use the Image package to open it up again
    plt.savefig(folder_name+'/Plots/Window_Plots/'+f'Window_{iteration_number}.png')

    # image = Image.open(analysis_path+"/"+date_path+"/"+"Plots/Window_Spectrograms/"+f'Window_{iteration_number}.png')
    # # image = Image.fromarray(data)
    # # image = image.convert('RGB')
    # # show PIL image
    # im_file = BytesIO()
    # img_save = image.save(im_file, format='PNG')
    # # img_save = image.save(analysis_path+"/"+date_path+"/"+"Plots/"+im_file, format='PNG')
    # im_bytes = im_file.getvalue()

    # img_str = "data:image/png;base64," + base64.b64encode(im_bytes).decode()
    plt.close()
    # return img_str
    
for i in np.arange(stacked_windows.shape[0]):
    if i%10 == 0:
        print(f'Iteration {i} of {stacked_windows.shape[0]}')
    data = stacked_windows[i,:]
    window_times = stacked_window_times[i,:]
    embeddable_image(data, window_times, i)

from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, ColumnDataSource

output_file(filename=f'{folder_name}/Plots/umap.html')

digits_df = pd.DataFrame(embedding, columns=('x', 'y'))

# Create a ColumnDataSource from the data
source = ColumnDataSource(digits_df)

# Create a figure and add a scatter plot

p = figure(width=800, height=800, tools=('pan, wheel_zoom, hover, reset'))
p.scatter(x='x', y='y', source=source)


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
for i in np.arange(digits_df.shape[0]):
    source.data['image'].append(f'{folder_name}/Plots/Window_Plots/Window_{i}.png')

show(p)

save(p)


embedding = np.load(folder_name+'/UMAP_Embedding.npy')









# base64_str = embeddable_image(list_of_images[0])


# # save base64_str to a file
# with open('base64_str.txt', 'w') as f:
#     f.write(base64_str)

# img_str_list = []
# for i in np.arange(stacked_windows.shape[0]):
#     print(i)
#     img_str = embeddable_image(stacked_windows, masked_frequencies, stacked_window_times, i)
#     img_str_list.append(img_str)

# digits_df = pd.DataFrame(embedding, columns=('x', 'y'))
# masked_frequencies.shape = (masked_frequencies.shape[0], 1)

# image_list = []
# for i in np.arange(digits_df.shape[0]):
#     data = stacked_windows[i,:]
#     window_times = stacked_window_times[i,:]
#     stringval = embeddable_image(data, window_times, i)
#     image_list.append(i)

# # digits_df['image'] = list(map(embeddable_image, stacked_windows, stacked_window_times))
# datasource = ColumnDataSource(data = digits_df)

# plot_figure = figure(
#     title='UMAP',
#     width=1000,
#     height=1000,
#     tools=('pan, wheel_zoom, reset')
# )

# plot_figure.add_tools(HoverTool(tooltips="""
# """))

# plot_figure.circle(
#     'x',
#     'y',
#     source=datasource,
#     fill_color = 'gray'
# )

# save(plot_figure)

plt.figure()
plt.title(f'Number of Spectrograms: {num_spec}, Window Size: {window_size} Pixels, Stride Size: {stride} Pixels', fontsize = 24)
plt.plot(embedding[:,0], embedding[:,1], color = 'blue')
plt.show()
plt.savefig((folder_name+"/"+"Plots"+"/"+f'Transition_Trajectory.pdf'))

plt.close('all')












