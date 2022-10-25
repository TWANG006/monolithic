from audioop import rms
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
import math
from monolithic.math.statistics import rmse, vrr
from scipy import *
from scipy import signal

g = 1   #阻尼因子,值≤1时，g值越大收敛速度越快，而g值过大则迭代发散
s = 0.9  #s是g的修正系数,s的值最好是接近1的值

R =  # 单位时间内的去除函数
H0 = # 所需的去除物  R，D的二维卷积
dx =   # x,y:tif的区域
dy = 
VR = vrr(R,dx,dy) # 去除函数的体积去除率,确定初始停留时间和停留时间的修正量

def itea(E,H0,R,VR,D,g):
    k = E / VR
    D = D + g*k
    E = H0 - signal.convolve2d(R, D, 'full')

    return E,D

def ite(H0,R,VR,g,s):
    """迭代计算驻留时间"""
    H0 = H0 - np.min(H0)  #将H0处理为非负
    D0 = H0 / VR   #初始驻留时间
    E0 = H0 - signal.convolve2d(R, D0, 'full') #初始表面残差
    D = D0
    E = E0  
    k = 0  
    while k >= 0:
        if rmse(E,_= itea(E,H0,R,VR,D,g)) > rmse(E):
            if rmse(E)/rmse(E0) < 1:
               _,D = itea(E,H0,R,VR,D,g)
               break
            else:
                g = s*g
        else:
            E,D = itea(E,H0,R,VR,D,g)
            k =k +1           
    return D        
    



