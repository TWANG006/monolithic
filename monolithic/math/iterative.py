from audioop import rms
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
from monolithic.math.statistics import rmse, vrr,pv,rms
from scipy import *
from scipy import signal
from monolithic.plot import show_surface_map
from monolithic.math import remove_surface, remove_sphere, remove_polynomials
import copy
from mpl_toolkits.mplot3d import Axes3D
import time

# 产生一个低频二维正弦条纹
T = 5
f = 0.0303  #频率
N = 200 #采样点选择256
a = 70e-6  #单位mm
x = np.linspace(-20, 20, N)
y = np.linspace(-20, 20, N)
X,Y = np.meshgrid(x,y) #生成一个矩阵，并填充数据.
Z_low = a*np.sin(2 * np.pi * f * X) 


# 产生一个高频二维正弦条纹
T = 5
f = 8.4  #频率
N = 200 #采样点选择256
a = 70e-6  #单位mm
x = np.linspace(-20, 20, N)
y = np.linspace(-20, 20, N)
X,Y = np.meshgrid(x,y) #生成一个矩阵，并填充数据.
Z_high = a*np.sin(2 * np.pi * f * X) 


# 产生一个高频+低频二维正弦条纹
T = 5
f1 = 8.4  #频率
f2 = 0.0303
N = 200 #采样点选择256
a = 70e-6  #单位mm
x = np.linspace(-20, 20, N)
y = np.linspace(-20, 20, N)
X,Y = np.meshgrid(x,y) #生成一个矩阵，并填充数据.
Z_high = a*np.sin(2 * np.pi * f1 * X) 
Z_low = a*np.sin(2 * np.pi * f2 * X) 
Z = Z_high + Z_low

# R
x_r = np.linspace(-2, 2, 21) 
y_r = np.linspace(-2, 2, 21)
X_r,Y_r = np.meshgrid(x_r, y_r)
w0 = 1  #振幅
R = np.exp(-((pow(X_r, 2) + pow(Y_r, 2)) / pow(w0, 2)))


H0 =  Z #初始面型
g = 1  #松弛因子ξ，初始值为1。利用下面的s来修正松弛因子。
s = 0.95 #松弛因子每次递减量。松弛因子修正系数ψ.
dx = 21  # x,y:tif的区域  
dy = 21
VR =  vrr(R,dx,dy) # 去除函数的体积去除率,确定初始停留时间和停留时间的修正量


def iterative_1(E,H0,R,VR,D,g):
    deltak = E / VR  #：Δ_k为本次迭代的驻留时间校正值
    D = D + g*deltak #总的驻留时间矩阵：上次迭代时的驻留时间+本次迭代的驻留时间校正值*松弛因子
    D[D<0]=0  #将D处理为非负
    E = H0 - signal.convolve2d(D, R, 'same') #残差
    return E,D

# R:单位时间内的去除函数  H0:所需的去除物  H为R，D的二维卷积
# g阻尼因子,值≤1时，g值越大收敛速度越快，而g值过大则迭代发散;s是g的修正系数,s的值最好是接近1的值
def iterative(H0,R,VR,g,s):
    """迭代计算驻留时间
    Args:
        H0 (numpy.ndarray): the input `numpy` array of data.
        R (numpy.ndarray): the input `numpy` array of data. Extracted TIF. 
        VR (numpy.ndarray): the input `numpy` array of data. Calculate the removal intensity. 
    Summing up the removal amount in each unit grid area is actually calculating the volume removal amount. 
        g (float): the relaxation factor ξ, with an initial value of 1. Use the following yita to correct the relaxation factor.
        s (float): the amount of each decrement of the relaxation factor. Relaxation factor correction factor ψ.

    Returns:
        E: the participation Error of all the element in x.
        D: the dwell time of all the element in x.
    """
    H0 = H0 - np.nanmin(H0) #将H0处理为非负,它还能保证获得的驻留时间是非负的
    D0 = H0 / VR   #初始驻留时间
    E0 = H0 - signal.convolve2d(D0, R, 'same') #初始表面残差
    D = D0
    E = E0
    k = 0
    a = np.fix(len(R)/4).astype(int)
    while 1:
        E0 = E[a:-a,a:-a]
        E1 = iterative_1(E,H0,R,VR,D,g)[0][a:-a,a:-a]
        if rms(E1) / rms(E0) >= 0.9999:  #若残差收敛比接近于1，则终止迭代
            E,D = iterative_1(E,H0,R,VR,D,g)
            break
        else : 
            if rms(E0) < rms(E1) :  #迭代后的RMS比上一次RMS差
                if rms(E0)/rms(H0) < 0.1:  #如果残差收敛比小于0.1，则终止迭代
                    E,D = iterative_1(E,H0,R,VR,D,g)
                    break
                else:
                    g = s*g  #若RMS没有初始RMS的10倍，则更换松弛因子，继续计算
                    E,D = iterative_1(E,H0,R,VR,D,g)                     
            else:  #若迭代后的RMS比上一次的要好就继续迭代
                E,D = iterative_1(E,H0,R,VR,D,g)  
                k =k + 1  
        
    return E,D 

E,D = iterative(H0,R,VR,g,s)

