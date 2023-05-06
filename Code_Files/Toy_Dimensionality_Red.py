#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:42:49 2023

@author: ananyakapoor
"""

import scipy.io
import matplotlib.pyplot as plt
mat = scipy.io.loadmat('/Users/ananyakapoor/Downloads/phonenumbers.mat')

import numpy as np
import sounddevice as sd

signal = mat['signal']
fs = 4096
# sd.play(signal, fs)

signal_spec = signal.copy()
signal_spec.shape = (signal.shape[1],)

# Varying the NFFT parameter 
NFFT_vec = np.arange(1000, 10000, 100)

for nfft_val in NFFT_vec:
    plt.clf();
    plt.ion()
    spectrum, freqs, t, im = plt.specgram(signal_spec,   NFFT=nfft_val, Fs=fs, noverlap=120,cmap='jet')
    plt.title(f'NFFT Value: {nfft_val}, NOverlap = 1000')
    plt.xlabel("Time (t)")
    plt.ylabel("Frequency (omega)")
    # plt.legend()
    plt.draw();
    plt.pause(0.05);
    
# Varying the NOverlap parameter 
N_overlap = np.arange(1, 15, 1)

for n_overlap_val in N_overlap:
    plt.clf();
    plt.ion()
    spectrum, freqs, t, im = plt.specgram(signal_spec,   NFFT=128, Fs=fs, noverlap=n_overlap_val,cmap='jet')
    plt.title(f'NFFT Value: 128, NOverlap = {n_overlap_val}')
    # plt.legend()
    plt.draw();
    plt.pause(0.05);
    












spectrum, freqs, t, im = plt.specgram(signal_spec,   NFFT=5000, Fs=fs, noverlap=2,cmap='jet')

# Let's reshape the spectrogram so that time is the first dimension

# spect_reshaped = np.transpose(spectrum)

# Let's do a simple PCA
# First let's scale our data
# from sklearn.preprocessing import StandardScaler
# scaled_spec = StandardScaler().fit_transform(spectrum)

# from sklearn.decomposition import PCA

# pca = PCA(n_components=6)
# pca.fit(spectrum.T)
# X_pca = pca.transform(spectrum.T)


