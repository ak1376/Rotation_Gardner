#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:38:22 2023

@author: ananyakapoor
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

# Define domain from -pi to pi and stepping by 0.001
dx = 0.001
L = np.pi # Length of our domain?
x = L * np.arange(-1+dx,1+dx,dx) # -pi+0.001 to pi+0.001, stepping by 0.001. Tells us where we are in the time series
n = len(x)
nquart = int(np.floor(n/4)) # Used to create our pointy function

# Define hat function
f = np.zeros_like(x)
f[nquart:2*nquart] = (4/n)*np.arange(1,nquart+1)
f[2*nquart:3*nquart] = np.ones(nquart) - (4/n)*np.arange(0,nquart)

fig, ax = plt.subplots()
ax.plot(x,f,'-',color='k')

# Compute Fourier series
name = "Accent"
cmap = get_cmap('tab10')
colors = cmap.colors
ax.set_prop_cycle(color=colors)

A0 = np.sum(f * np.ones_like(x)) * dx # How did he get this?
fFS = A0/2 # Approximated function value. For k = 0 this is A0/2. Why? 

A = np.zeros(200) # Coefficient value for cosines
B = np.zeros(200) # Coefficient value for sines
for k in range(200):
    A[k] = np.sum(f * np.cos(np.pi*(k+1)*x/L)) * dx # Inner product
    B[k] = np.sum(f * np.sin(np.pi*(k+1)*x/L)) * dx
    fFS = fFS + A[k]*np.cos((k+1)*np.pi*x/L) + B[k]*np.sin((k+1)*np.pi*x/L)
    ax.plot(x,fFS,'-')