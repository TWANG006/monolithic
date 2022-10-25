from cmath import phase
from dataclasses import dataclass
from plistlib import FMT_XML
from turtle import end_fill
import numpy as np
import scipy.optimize.optimize
from matplotlib import pyplot as plt
from scipy.interpolate import interpn
from scipy.interpolate import griddata
import scipy.io as sio
from scipy import interpolate
import pandas as pd
from numpy import size
from numpy import squeeze
from matplotlib import cm
import copy
import re
from scipy.fftpack import fft
from sklearn import preprocessing
import operator
from functools import reduce
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
from scipy.optimize import linprog
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import struct
import sys
from monolithic.io import read_zygo_binary, read_zygo_dat, read_zygo_datx
from monolithic.plot import show_surface_map
from monolithic.math import remove_surface, remove_sphere, remove_polynomials
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture 
import time

p = 5
t = 12
r = 6e-3 #6e-3  4e-3 2.5e-3
thrd = 0

data = read_zygo_dat('E:\code\monolithic\data\R40-3-0.5mm.dat') #R40-3-0.5mm.dat R20-2-0.35mm.dat R10-2-0.28mm.dat
surfMpp = data['meta']['lateral_res']
Ztif1 = data['phase']

m,n = Ztif1.shape
Xtif1,Ytif1 = np.meshgrid(np.arange(0,n),np.arange(0,m))
Xtif1 = Xtif1*surfMpp
Ytif1 = Ytif1*surfMpp
Xtif1 = Xtif1[::p,::p] 
Ytif1 = Ytif1[::p,::p] 
Ztif1 = Ztif1[::p,::p] 

Ztif1,_,_,_ = remove_surface(Xtif1,Ytif1,Ztif1)
surfMpp = np.median(np.diff(Xtif1[0,:]))

# 初始面型
fig, ax = plt.subplots(1, 2, figsize=(10,10))
show_surface_map(Xtif1, Ytif1, Ztif1,ax = ax[0], title='Extracted TIF')

m,n = Ztif1.shape
center_x = fix(n/2)+1
center_y = fix(m/2)+1
hf_win_sz = fix(np.min([m,n])/2)-1
Ztif1 = Ztif1[(center_y - hf_win_sz).astype(np.int16) - 1: (center_y + hf_win_sz).astype(np.int16) ,
              (center_x - hf_win_sz).astype(np.int16) - 1: (center_x + hf_win_sz).astype(np.int16) ]

# fill NAN
Xtif1,Ytif1 = np.meshgrid(np.arange(-hf_win_sz,hf_win_sz+1),np.arange(-hf_win_sz,hf_win_sz+1))
Ytif1 = -Ytif1
Xtif1 = Xtif1*surfMpp
Ytif1 = Ytif1*surfMpp

#Remove tilt
id = Xtif1**2 +Ytif1**2 > r**2 
_,_,_,f = remove_surface(Xtif1[id],Ytif1[id],Ztif1[id]) #Xtif1.shape:328260
Ztif1 = Ztif1 - (f[0] + f[1]*Xtif1 + f[2]*Ytif1)

# show_surface_map(Xtif1, Ytif1, Ztif1,title='Footprint')

Ztif1[Ztif1>thrd] = thrd
Ztif1 = nanmax(Ztif1) - Ztif1
Ztif1 =Ztif1/t

r = 6.5e-3
pr = (hf_win_sz - fix(r / surfMpp)).astype(int)
Xtif1 = Xtif1[pr+1:-pr+1,pr+1:-pr+1]
Ytif1 = Ytif1[pr+1:-pr+1,pr+1:-pr+1]
Ztif1 = Ztif1[pr+1:-pr+1,pr+1:-pr+1]

# 去除面型
show_surface_map(Xtif1,Ytif1,Ztif1,ax = ax[1], title='Extracted TIF')
plt.tight_layout()


fig=plt.figure()
ax = Axes3D(fig)
ax.plot_surface(Xtif1, Ytif1, Ztif1,cmap='rainbow', rstride=1, cstride=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title('Extracted TIF')


# 拟合 gmm
# 2. Calculate VRR
# Ztif1[np.isnan(Ztif1)] = 0
# VRR = np.sum(Ztif1)

# # 3. Use TIF and VRR to generate a histogram to approximate the Probability Density Function (PDF)
# normZtif = Ztif1 / VRR  # normalize Ztif

# # counts = np.round(100/np.max(normZtif)).astype(int) # maximum
# counts = 250000
# histZtif = np.round(counts*normZtif).astype(int)

# dx = abs(np.median(np.diff(Xtif1[1, :])))
# dy = abs(np.median(np.diff(Ytif1[:, 1])))

# x = []
# y = []
# m, n = Ztif1.shape
# for i in range(m):
#     for j in range(n):
#         if histZtif[i,j] > 0:
#             x.extend((np.random.rand(histZtif[i,j])*0.5*dx + Xtif1[i,j] - 0.5*dx).tolist())
#             y.extend((np.random.rand(histZtif[i,j])*0.5*dy + Ytif1[i,j] - 0.5*dy).tolist())
            

# X = np.vstack((x, y)).T
# plt.scatter(X[:, 0], X[:, 1], s=5)

# # 5. Fit the GMM model
# Xtest = np.vstack((Xtif1.ravel(), Ytif1.ravel())).T

# gmm = GaussianMixture(
#     n_components=15, covariance_type="full", tol = 1e-6, 
#     max_iter=1000, reg_covar=0).fit(X)

# log_probs = gmm.score_samples(Xtest)

# Zfit = np.exp(log_probs).reshape(Ztif1.shape)*VRR*dx*dy
# Zfit -= np.min(Zfit)
# Zdif = Ztif1 - Zfit
# Zdif = Zdif - np.nanmean(Zdif)

def tif_gmm(
    data: tuple,
    counts: int = None,
    n_components: int = 15,
    max_iters: int = 1000, #500
    tol: float = 1e-6,
    reg_convar=0, #1e-11
) -> tuple:
    """
    Fit the TIF with Gaussian Mixture Model (GMM)

    Parameters
    ----------
        data: `tuple`, 
            (x, z) for 1d or (x, y, z) for 2d
        counts: `int`
            Number of points used for the GMM, if 'None', 
            the maximum one is 20
        n_components: `int`
            The number of Gaussian functions to use in GMM
        max_iters: `int`
            The number of maxmum iterations in the optimization
        tol: `float`
            The tolerence for the log likelyhood function
        reg_convar: `float`
            The small constant that will be multipled to 
            the covariance matrix for stability
    
    Returns
    -------
        gmm:
            The sklearn's gaussian mixture model
        zfit: `np.ndarray`
            The fitted TIF
    """
    # initialization
    t = time.perf_counter()
    dim = len(data) - 1
    if dim == 1:
        dx = np.median(np.diff(data[0]))
    elif dim == 2:
        dx = (
            abs(np.median(np.diff(data[0][0, :]))),
            abs(np.median(np.diff(data[1][:, 0])))
        )
    else:
        ValueError('Dimension of the data should be either 1D or 2D.')
    
    # calculate histogram
    vrr = np.sum(data[-1])  # vrr of the TIF
    normZtif = data[-1]/vrr  # normalize the TIF with vrr
    if counts is None:
        counts = np.round(20/np.max(normZtif)).astype(int)
    histZtif = np.round(counts*normZtif).astype(int)  # histogram
    
    # sampling each bin of the histogram
    x = [[] for i in range(dim)]
    
    # print(len(dx))
    # for i, histCount in enumerate(histZtif.ravel()):
    #     if histCount > 0:
    #         for j in range(len(x)):
    #             x[j].extend((np.random.rand(histCount)*0.5*dx[j] + data[j].ravel()[i] - 0.5*dx[j]).tolist())
    x = []
    y = []
    m, n = Ztif1.shape
    for i in range(m):
        for j in range(n):
            if histZtif[i,j] > 0:
                x.extend((np.random.rand(histZtif[i,j])*0.5*dx[0] + Xtif1[i,j] - 0.5*dx[0]).tolist())
                y.extend((np.random.rand(histZtif[i,j])*0.5*dx[1] + Ytif1[i,j] - 0.5*dx[1]).tolist())
            

    X = np.vstack((x, y)).T
    
    
    # fit the gmm
    # X = np.asarray(x).T
    # print('X:',X)
    gmm = GaussianMixture(
        n_components=n_components, 
        max_iter=max_iters, 
        tol=tol,
        reg_covar=reg_convar
    ).fit(X)

    # generate the fit TIF
    Xtest = np.asarray([data[i].ravel() for i in range(dim)]).T
    zfit = np.exp(gmm.score_samples(Xtest)).reshape(data[-1].shape)*vrr*np.prod(dx)
    zfit -= np.min(zfit)

    return gmm, zfit

gmm, zfit = tif_gmm(
    (Xtif1, Ytif1, Ztif1),
    counts=250000,
    n_components=10
)
t = time.perf_counter()
tif_gmm(
    (Xtif1, Ytif1, Ztif1),
    counts=250000,
    n_components=10
)
print(f'coast:{time.perf_counter() - t:.8f}s')

Zdif = Ztif1 - zfit
Zdif = Zdif - np.nanmean(Zdif)

# 6. Compare the measured and fitted TIF
fig, ax = plt.subplots(1, 3, figsize=(18,18))
show_surface_map(Xtif1, Ytif1, Ztif1, ax=ax[0], title='measured')
show_surface_map(Xtif1, Ytif1, zfit, ax=ax[1], title='fitted')
show_surface_map(Xtif1, Ytif1, Zdif, vmin=-10, vmax=10, ax=ax[2], title='residual')

fig=plt.figure()
ax = Axes3D(fig)
ax.plot_surface(Xtif1, Ytif1, np.abs(Zdif),cmap='rainbow',rstride=1, cstride=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title('fitted')
plt.show()





