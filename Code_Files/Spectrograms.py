#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:20:34 2023

@author: ananyakapoor
"""

import scipy.io
import matplotlib.pyplot as plt
mat = scipy.io.loadmat('/Users/ananyakapoor/Downloads/phonenumbers.mat')

import numpy as np
import sounddevice as sd

signal = mat['signal']
fs = 4096
sd.play(signal.T, fs)

signal_spec = signal.copy()
signal_spec.shape = (signal.shape[1],)

spectrum, freqs, t, im = plt.specgram(signal_spec, NFFT=128, Fs=fs, noverlap=120,cmap='jet')


plt.figure()
plt.specgram(signal_spec, NFFT=128, Fs=fs, noverlap=120,cmap='jet')
plt.colorbar()
plt.show()

# =============================================================================
# # Spectrogram of Bach's Invention in F Major
# =============================================================================

import numpy as np
from pydub import AudioSegment
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# Load MP3 file
audio_file = AudioSegment.from_file("/Users/ananyakapoor/Downloads/J.S. Bach_ 15 Inventions, BWV 772-786 - No. 8 in F Major, BWV 779 (320 kbps).mp3", format="mp3")

# Convert to raw audio data
raw_data = audio_file.raw_data

# Convert to NumPy array
audio_array = np.frombuffer(raw_data, dtype=np.int16)

# # Normalize between -1 and 1
# audio_array = audio_array / 2**15

plt.figure()
plt.specgram(audio_array[1:1000000], NFFT=128,noverlap=120, Fs=44100,cmap='jet')
plt.colorbar()
plt.xlabel("Time (t)")
plt.ylabel("Frequency (omega)")
plt.show()








