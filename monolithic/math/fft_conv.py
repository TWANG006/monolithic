
""" fft_1d,fft_2d"""

import numpy as np
from scipy.fftpack import fft
from numpy.fft import fft,ifft,fft2,ifft2


def fast_zero_padding_1d(f, m):
    """
Purpose:
    Fast zero-padding of f0 to thse length m, m > length(f0)
Inputs: 
    f0: vector to be padded
    m: padding size
Outputs:
    f: padded vector
Info:
    Author: tianyiwang666@gmail.com (Dr WANG Tianyi)
    Copyright reserved
    """
    f0 = f0[:]  # column vector
    mf0 = min(f0.shape)  # lenght of the f0

    # generate a larger vector with m elements
    f = np.zeros(m, 1)

    # copy the original data
    f[0: mf0] = f0

    return f

def conv_fft_1d_crop(g, mf, mk):
    """
    Purpose:
    crop the correct portion to recover the same size of f
    """  
    if np.mod(mk, 2) == 1: 
        hmk = (mk-1)/2
    else:
        hmk = mk/2

    gCrop = g[hmk: mf+hmk, 1]

    return gCrop
  

def conv_fft_1d(f, k):
    """
Purpose:
    Implement 1D convolution using FFT whose result coincide with conv
    'same'
Inputs:
    f: signal
    k: kernel
Outputs:
    g: convolved signal
Info:
    Author: tianyiwang666@gmail.com (Dr WANG Tianyi)
    Copyright reserved
    """
    f = f[:]
    k = k[:]

    #  Compute the FFT padding size
    mf = min(f.shape)   # signal length
    mk = min(k.shape)   # kernel length
    m = mf + mk - 1      # padding length

    # Padding the f and k
    f = fast_zero_padding_1d(f, m)
    k = fast_zero_padding_1d(k, m)

    # Perform FFT & convolution
    g = np.real(ifft(np.multiply(fft(f),fft(k))))

    # Crop the correct portion to recover the "same" size of f
    g = conv_fft_1d_crop(g, mf, mk)

    return g

def fast_zero_padding_2d(F0,mm,nn):
    """
    expand f0 to [m n]
    this function can be realized by padarray, but is slower
    """
    # size f0
    m,n = F0.shape

    # generate a larger matrix with size [mm nn]
    F=np.zeros(mm,nn)

    # copy original data
    F[0:m,0:n]=F0

    return F

def conv_fft_2d_crop(R_crop, mS, nS, mK, nK):
    """  Crop the correct portion to recover the same size of S """
    if np.mod(mK,2) == 1 & np.mod(nK,2) == 1:
        hmK = (mK - 1)/ 2
        hnK = (nK - 1)/ 2
        R_crop = R_crop[hmK:mS+hmK,hnK:nS+hnK]
    elif np.mod(mK,2) == 0 & np.mod(nK,2) == 1:
        hmK = mK / 2
        hnK = (nK - 1)/ 2
        R_crop = R_crop[hmK:mS+hmK,hnK:nS+hnK]
    elif np.mod(mK,2) == 1 & np.mod(nK,2) == 0:
        hmK = (mK - 1) / 2
        hnK = nK / 2
        R_crop = R_crop[hmK:mS+hmK,hnK:nS+hnK]
    else:
        hmK = mK/2
        hnK = nK/2
        R_crop = R_crop[hmK:mS+hmK,hnK:nS+hnK]

    return R_crop


def conv_fft_2d(S,K):
    """
Purpose:
	Implement 2D convolution using FFT whose results conincide with conv2 'same'

Inputs:
   S: signal
   K: kernel

Output:
   R: convolved signal

Info:
   Contact: tianyiwang666@gmail.com (Dr WANG Tianyi)
   Copyright reserved.
   """
    # Signal size
    mS,nS = S.shape

    # Kernal size
    mK,nK = K.shape

    #  Padding size
    m = mS + mK - 1
    n = nS + nK - 1

    #  Padding R and B
    S = fast_zero_padding_2d(S, m, n)
    K = fast_zero_padding_2d(K, m, n)

    #  Perform FFT & convolution
    R = np.real(ifft2(np.multiply(fft2(S),fft2(K))))

    #  Crop the correct portion to recover the same size of S
    R = conv_fft_2d_crop(R, mS, nS, mK, nK)

    return R




