from audioop import rms
from time import process_time_ns
from tkinter import E
import numpy as np
import scipy.optimize.optimize
from matplotlib import pyplot as plt
from scipy.interpolate import interpn
from scipy.interpolate import griddata
import scipy.io as sio
import math
from monolithic.math.statistics import rmse, vrr,pv
from scipy import *
from scipy import signal
from monolithic.math import remove_surface, remove_sphere, remove_polynomials
import copy
from mpl_toolkits.mplot3d import Axes3D

surfFile = 'E:\\Codes\Multi-Tool\\1-initial-map\\data\\initial_map_rectangular_pap_60mm.mat'
readData = sio.loadmat(surfFile) # 读出readData的内容
Z = readData['Z']
plt.pcolormesh(Z)
plt.show()


g = 0.1   #阻尼因子,值≤1时，g值越大收敛速度越快，而g值过大则迭代发散
s = 0.95  #s是g的修正系数,s的值最好是接近1的值

# R:单位时间内的去除函数  H0:所需的去除物  R，D的二维卷积
R = np.full(Z.shape,0.000015)#np.array([[0.000015]])  


H0 =  Z
dx = 20  # x,y:tif的区域  
dy = 20
VR = 1 #vrr(R,dx,dy) # 去除函数的体积去除率,确定初始停留时间和停留时间的修正量
print('VR:',VR)
# print(pv(Z))
def rms(x: np.ndarray) -> float:
    """Calculate the RMS of all the element in the input data.

    Args:
        x (numpy.ndarray): the input `numpy` array of data.

    Returns:
        float: the RMS error of all the element in x.
    """
    return np.sqrt(np.nanmean(np.square(x.ravel())))


def itea(E,H0,R,VR,D,g):
    k = E / VR
    # print('k:',np.nanmin(k))
    D = D + g*k
    # print('it_D:',D)
    # print('ite_D_m:',np.nanmin(D))
    E = H0 - signal.convolve2d(R, D, 'same')
    # print('ite_H_m:',np.nanmean(H0))
    # print('si_m:',np.nanmean(signal.convolve2d(R, D, 'same'))) #算术平均值
    # print('ite_e_m:',np.nanmean(D))
    a = np.fix(len(E)/4).astype(int)
    print('rms_e:',rms(E[a:-a,a:-a]))
    return E,D

def ite(H0,R,VR,g,s):
    """迭代计算驻留时间"""
    # print(np.nanmin(H0))
    H0 = H0 - np.nanmin(H0) #将H0处理为非负,它还能保证获得的驻留时间是非负的
    print(H0)
    D0 = H0 / VR   #初始驻留时间
    E0 = H0 - signal.convolve2d(R, D0, 'same') #初始表面残差
    print('rm_e:',rms(E0))
    D = D0
    E = E0
    
    k = 0
    while k >= 0:
        
        # print('rms_ek:',rms(itea(E,H0,R,VR,D,g)[0]))
        # print(int(len(E)/4))
        a = np.fix(len(E)/4).astype(int)
        print('a:',a)
        # print(a)
        # E = E[a:-a,a:-a]
        # print('E:',E.shape)
        if rms(E[a:-a,a:-a]) < rms(itea(E,H0,R,VR,D,g)[0][a:-a,a:-a]):
            if rms(E[a:-a,a:-a])/rms(E0[a:-a,a:-a]) < 1:
                _,D = itea(E,H0,R,VR,D,g)
                print(1111111111)
                break
            else:
                g = s*g
                E,D = itea(E,H0,R,VR,D,g)
                print(222222222222222222222)
        else:
            E,D = itea(E,H0,R,VR,D,g)
            print(33333333333)
            k =k +1           
    return D  

fig=plt.figure()
plt.imshow(H0,vmin=0,vmax=255)

plt.show()

D = ite(H0,R,VR,g,s)
print('D:',D)

# X1 = np.array([[2,4],[6,8]])
# X2 = np.array([[4,5],[7,9]])
# a = np.nanmean(X1)
# b = np.nanmean(X2)
# c = rms(X1)
# d = rms(X2)
# print(a)
# print(b)
# print(c)
# print(d)
