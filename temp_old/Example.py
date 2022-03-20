# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 11:05:29 2021

@author: dagpa
"""
# =============================================================================
# EXAMPLE pyOMA
# =============================================================================
# Import modules
import PyOMA as OMA
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


import scipy.io
mat = scipy.io.loadmat('Before.mat')


# open the file with pandas and create a dataframe 
data = pd.read_csv("data.txt", header=None, sep="\t", index_col=False) 
data = np.array(data)

# Sampling frequency
fs = 833 # [Hz] Sampling Frequency
q = 10 # Decimation factor

# Detrend and decimate
data = signal.detrend(data, axis=0) # Rimozione trend
data = signal.decimate(data,  q, ftype='fir', axis=0) # Decimazione segnale
fs = fs/q # [Hz] Decimated sampling frequency

# Run FDD
FDD = FDDsvp(data,  fs)

# Define list/array with the peaks identified from the plot
FreQ = [4.165] # identified peaks

# Extract the modal properties 
Res_FDD = FDDmodEX(FreQ, FDD[1])
Res_EFDD = EFDDmodEX(FreQ, FDD[1], method='EFDD')
Res_FSDD = EFDDmodEX(FreQ, FDD[1], method='FSDD')

# Run SSI
br = 15
SSIcov= SSIcovStaDiag(data, fs, br, ordmax=25)
SSIdat = SSIdatStaDiag(data, fs, br, ordmax=25) 

# Extract the modal properties
Res_SSIcov = SSIModEX(FreQ, SSIcov[1],deltaf=0.01)
Res_SSIdat= SSIModEX(FreQ, SSIdat[1],deltaf=0.01)


# =============================================================================
# 
# =============================================================================

_MS_EFDD = Res_EFDD['Mode Shapes']
_MS_FSDD = Res_FSDD['Mode Shapes']
_MS_SSIcov = Res_SSIcov['Mode Shapes']
_MS_SSIdat = Res_SSIdat['Mode Shapes']
_nch = data.shape[1]

MACmatr = np.reshape(
        [MaC(FInorm_1[:,_l],_MS_SSIcov[:,_k].real) for _k in range(_nch) for _l in range(_nch)], # (_nch*_nch) list of MAC values 
        (_nch,_nch)) # new (real) shape (_nch x _nch) of the MAC matrix

autoMAC = np.reshape(
        [MaC(_MS_SSIcov[:,_l].real,_MS_SSIdat[:,_k].real) for _k in range(_nch) for _l in range(_nch)], # (_nch*_nch) list of MAC values 
        (_nch,_nch)) # new (real) shape (_nch x _nch) of the MAC matrix

# PLOTTO MATRICE MAC 
meas = ["mode I", "mode II", "mode III", "mode IV", "mode V"]
num = ["mode I", "mode II", "mode III", "mode IV", "mode V"]

fig, ax = plt.subplots()
im, cbar = heatmap(MACmatr*100, num, meas, ax=ax,
                   cmap="jet", cbarlabel="MAC [%]")

texts = annotate_heatmap(im, valfmt="{x:.2f}")

fig.tight_layout()
plt.show()
