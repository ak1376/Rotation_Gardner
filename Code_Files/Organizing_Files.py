#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 12:04:18 2023

@author: ananyakapoor
"""
import os
import shutil
import numpy as np

directory = '/Users/ananyakapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_data_matrices'
all_files = os.listdir(directory)

folder_name = "MATLAB_Files"
try:
    os.mkdir(directory+'/'+folder_name)
    print(f"Folder '{folder_name}' created successfully!")
except FileExistsError:
    print(f"Folder '{folder_name}' already exists.")
except Exception as e:
    print(f"An error occurred while creating the folder: {e}")

source_folder = directory
destination_folder = directory+'/MATLAB_Files'

mat_files_to_move = [element for element in all_files if '.mat' in element]

for i in np.arange(len(mat_files_to_move)):
    destination_folder_file = destination_folder+'/'+mat_files_to_move[i]
    source_folder_file = source_folder+'/'+mat_files_to_move[i]
    # Move the file from the source folder to the destination folder
    try:
        shutil.move(source_folder_file, destination_folder_file)
        print(f"Moved '{mat_files_to_move[i]}' to '{destination_folder}' successfully!")
    except Exception as e:
        print(f"An error occurred while moving '{mat_files_to_move[i]}': {e}")