#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:46:28 2023

@author: ananyakapoor
"""

import scipy.io
import matplotlib.pyplot as plt
mat = scipy.io.loadmat('/Users/ananyakapoor/Downloads/phonenumbers.mat')

import numpy as np
import sounddevice as sd

signal = mat['signal']
fs = 4096
# # sd.play(signal, fs)

# # Let's do the FFT

n = signal.shape[1] # how many data points are in our signal
fhat = np.fft.fft(signal, n) # These are the complex-valued Fourier coefficients with a magnitude (real) and phase (imaginary)
PSD = fhat*np.conj(fhat)/n # Computing the power spectral density : power per frequency
dt = (1/4096)

freq = (1/(dt*n))*np.arange(n)
L = np.arange(1, np.floor(n/2), dtype = 'int')
PSD.shape = (n, 1)
freq.shape = PSD.shape

plt.figure()
plt.plot(freq[L], PSD[L])
plt.xlabel('Frequency')
plt.ylabel("Power")
plt.title("Fourier Transformed Representation")

dt = 1 / fs  # sampling interval
freqs = np.fft.fftfreq(n, dt)
magnitudes = np.abs(fhat)

mask = np.ones_like(magnitudes)
freqs.shape = mask.shape
frequencies_to_filter = freqs<400
frequencies_to_filter.shape = (1, n)
mask[frequencies_to_filter] = 0
filtered_fft = fhat * mask
filtered_signal = np.fft.ifft(filtered_fft)
filtered_signal = filtered_signal.real
sd.play(filtered_signal.T, fs)
    
    
    