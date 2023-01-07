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

<<<<<<< Updated upstream

=======
"""初代迭代驻留时间算法"""
>>>>>>> Stashed changes
def iterative_1(E,H0,R,VR,D,g):
    deltak = E / VR  #：Δ_k为本次迭代的驻留时间校正值
    D = D + g*deltak #总的驻留时间矩阵：上次迭代时的驻留时间+本次迭代的驻留时间校正值*松弛因子
    D[D<0]=0  #将D处理为非负
    E = H0 - signal.convolve2d(D, R, 'same') #残差
    return E,D

# R:单位时间内的去除函数  H0:所需的去除物  H为R，D的二维卷积
# g阻尼因子,值≤1时，g值越大收敛速度越快，而g值过大则迭代发散;s是g的修正系数,s的值最好是接近1的值
<<<<<<< Updated upstream
def iterative(H0,R,VR,g,s):
=======
def iterative_method(H0,R,VR,g,s):
>>>>>>> Stashed changes
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
        EE0 = E[a:-a,a:-a]
        EE1 = iterative_1(E,H0,R,VR,D,g)[0][a:-a,a:-a]
        if rms(EE1) / rms(EE0) >= 0.9999:  #若残差收敛比接近于1，则终止迭代
            E,D = iterative_1(E,H0,R,VR,D,g)
            break
        else : 
            if rms(EE0) < rms(EE1) :  #迭代后的RMS比上一次RMS差
                if rms(EE0)/rms(E0) < 0.1:  #如果残差收敛比小于0.1，则终止迭代
                    E,D = iterative_1(E,H0,R,VR,D,g)
                    break
                else:
                    g = s*g  #若RMS没有初始RMS的10倍，则更换松弛因子，继续计算
                    E,D = iterative_1(E,H0,R,VR,D,g)                     
            else:  #若迭代后的RMS比上一次的要好就继续迭代
                E,D = iterative_1(E,H0,R,VR,D,g)  
                k =k + 1  
        
    return E,D 

<<<<<<< Updated upstream
E,D = iterative(H0,R,VR,g,s)

=======
"""改进的迭代驻留时间算法"""
def iterative(Z_residual,Z,Ztif,Vrr,D,kxi):
    """
    Z_residual: the residual of the clear aperture
    Z: the initial height error [m]
    Ztif: the height of the TIF [m/s]
    vrr: volumetric removal rate of the TIF [m/s]
    D: the dwell time of all the element in Z
    kxi: the relaxation factor ξ. Use the following yita to correct the relaxation factor
    Purpose:
        Calculating residuals and dwelltime.
    Returns:
        Zca_residual: the residual of the clear aperture
    D: the dwell time of all the element in Z
    Referecen:
        Wang C, Yang W, Wang Z, Yang X, Hu C, Zhong B, Guo Y, Xu Q. Dwell-time algorithm for polishing large optics.
        Appl Opt. 2014 Jul 20;53(21):4752-60. doi: 10.1364/AO.53.004752. PMID: 25090214.
    Author:
        Qingqing Cui (2226773129@qq.com)

    """
    deltak = Z_residual / vrr #Δ_k为本次迭代的驻留时间校正值
    D = D + kxi*deltak #总的驻留时间矩阵：上次迭代时的驻留时间+本次迭代的驻留时间校正值*松弛因子
    D[D<0]=0 #将D处理为非负
    Z_residual = Z - signal.convolve2d(D, Ztif, 'same')#残差
    return Z_residual,D

def dwell_time_iterative_method(Z,Ztif,sx,sy,kxi,psi,k,FWHM):
    """
    Z: the initial height error [m]
    Ztif: the height of the TIF [m/s]
    sx,sy: the unit grid length,and the unit grid width [mm]
    kxi: the relaxation factor ξ. Use the following yita to correct the relaxation factor.
    psi: the amount of each decrement of the relaxation factor. Relaxation factor correction factor ψ.
    k: the number of iterations
    FWHM: the full width at half maxima

    Purpose:
        Calculating the dwell time using the iterative method.
    Returns:
        Zca_residual: the residual of the clear aperture
    D: the dwell time of all the element in Z
    Referecen:
        Wang C, Yang W, Wang Z, Yang X, Hu C, Zhong B, Guo Y, Xu Q. Dwell-time algorithm for polishing large optics.
        Appl Opt. 2014 Jul 20;53(21):4752-60. doi: 10.1364/AO.53.004752. PMID: 25090214.
    Author:
        Qingqing Cui (2226773129@qq.com) 
    """
    
    V=Ztif*sx*sy #计算去除强度矩阵。去除量*单位网格长度*单位网格宽度，其实就是单位网格面积内的去除量
    vrr=np.sum(V[:])#计算去除强度。把各个单位网格面积内的去除量求和，其实就是计算体积去除量。

    Z = Z - np.nanmin(Z)#将H0处理为非负,它还能保证获得的驻留时间是非负的
    D0 = Z / vrr#初始驻留时间
    Z0_residual = Z - signal.convolve2d(D0, Ztif, 'same')#初始表面残差
    D = D0
    Z_residual = Z0_residual

    i = 0#用于限制迭代次数
    j = 0#用于限制迭代修复次数
    aa = np.fix(FWHM)#用于截取去除效果较好的区域
    [mx,my] = np.shape(Z_residual)

    #在迭代限制次数内，进行驻留时间算法
    while i<=k:
        Zz0_residual = Z_residual[aa+1:my-aa,aa+1:mx-aa]
        Z1_residual,D = iterative(Z_residual,Z,Ztif,vrr,D,kxi)
        Zz1_residual = Z1_residual[aa+1:my-aa,aa+1:mx-aa]
        rms_Z0_residual = rms(Zz0_residual)#计算本次残差的rms
        rms_Z1_residual = rms(Zz1_residual) #计算迭代后残差的rms
        
        if rms_Z0_residual<=10e-6: #如果残差的rms达到目标值，则直接退出循环
            break
        else:#残差的rms没有达到目标值
            if (rms_Z0_residual < rms_Z1_residual) and (rms_Z0_residual!=0 or rms_Z0_residual!=np.Nan): #下一次残差的rms比上一次残差的rms大，则进行修正
                while j<=10:#最多进行10次修正
                    Z0_residual = Z_residual[aa+1:my-aa,aa+1:mx-aa]
                    rms_Z0_residual = rms(Z0_residual)
                    kxi = psi*kxi#更换松弛因子，对残差进行修正。ξ=ψ*ξ
                    [Z_residual,D]=iterative(Z_residual,Z,Ztif,vrr,D,kxi)
                    Zz0_residual = Z_residual[aa+1:my-aa,aa+1:mx-aa]
                    rms_Zz0_residual = rms(Zz0_residual)
                    if rms_Z0_residual > rms_Zz0_residual:#如果残差被修正，也就是下一次残差的rms比上一次残差的rms小，则退出本循环
                        break
                    elif j==8:
                        [E1,D1]=iterative(Z_residual,Z,Ztif,vrr,D,kxi)
                    elif j==9:#若第10次循环，残差仍未被修正，下一次残差的rms比上一次残差的rms大，则返回上一次的残差
                        Z_residual = E1
                        D = D1
                    j = j+1  
            else:#若残差的rms不需要被修正，则使用本次的残差继续进行迭代
                [Z_residual,D]=iterative(Z_residual,Z,Ztif,vrr,D,kxi)
        
            i = i+1
    #返回截取后的残差
    Zca_residual = Z_residual[aa+1:my-aa,aa+1:mx-aa]
    return Zca_residual,D


""" 王春锦老师迭代驻留时间算法"""
def iterative(R,H_MRR,VR):
    Ini_RMS=rms(R) #：初始RMS
    T=R/VR #：初始面形R除以体积去除量B，得到的T就是初始驻留时间(对应公式5.3,D0=H0/VR)
    temp=signal.convolve2d(T,H_MRR,'same') #：二维卷积运算，temp为驻留后的面形
    E0=R-temp #：对应公式5.5，E_0=H_0-R**D_0，即初始驻留时间后的表面残余误差矩阵E0=初始面形R-初始驻留时间的去除
    ksi=1#松弛因子ξ，初始值为1。利用下面的yita来修正松弛因子。每循环一次，修正因子就依次变为0.95,0.95^2,0.95^3,0.95^4,0.95^5,...
    yita=0.95#松弛因子每次递减量。%_#：松弛因子修正系数ψ
    a = np.fix(len(R)/4).astype(int)
    while 1:
        succ=0#如果succ=1，则表示迭代成功
        kk=0#改动
        while 1:
            E_1=E0[a:-a,a:-a] #此次迭代前的中间部分 #：即扣掉两侧边缘的中间点数
            tempRMS_1=rms(E_1)#此次迭代前的中间部分RMS。%_#：初始的RMS
            deltak=E0/VR #：Δ_k为本次迭代的驻留时间校正值。对应公式5.6，Δ_k=E_k/VR。（这里，B就是体积去除量VR）
            TT=T+ksi*deltak #：总的驻留时间矩阵：上次迭代时的驻留时间+本次迭代的驻留时间校正值*松弛因子。对应公式5.7，D_k+1=D_k+ξ*Δ_k
            TT[TT<0]=0 #：把驻留时间TT中小于0的全设为0
            EE0=R-signal.convolve2d(TT,H_MRR,'same') #：EE0为本次迭代的残余误差，为初始面形H_0-本次驻留时间TT下的面形(即R**D_k+1)。对应E_k+1=H_0-R**D_k+1。
            E_2=EE0[a:-a,a:-a]#此次迭代后的中间部分
            tempRMS_2=rms(E_2) #此次迭代后的中间部分的RMS
            if (tempRMS_2 > tempRMS_1) & (tempRMS_1!=0): #若迭代后的RMS比上一次还差，则终止迭代
            #         T=T;%此行表示驻留时间就是用跟上次的驻留时间，可省去
                E=E_1#此行表示中间部分的残余误差就用上次的残余误差
            #         E0=E0;%此行表示整个面的参与误差就用上次的残余误差可省略
                if (Ini_RMS/tempRMS_1)>=10:  #对应流程图的RMSk/RMS0<η，（这里的η=10，可取其他数值）
                    succ=1#标注迭代成功
                    break
                else:
                    break  
            E0=EE0 #：如果迭代后的RMS比上一次好，那就需要继续迭代。则把迭代后的残余误差赋值给E0，把迭代后的驻留时间TT赋值给T，然后kk加1。
            T=TT #：kk数值表明了迭代次数，kk并没有参与上面的计算，实际参与运算的是残余误差E0和驻留时间矩阵TT。
            kk=kk+1#可修改为迭代的后一步残余误差若大于上一部的残余误差，则停止迭代。并考虑如何限定驻留总时间     
        if succ==1:
            break
        ksi=ksi*yita #：如果succ==0，则说明RMS没有初始MRS的10倍，则更换松弛因子，继续计算。ξ=ψ*ξ。  


    return E0,TT,kk 

E,D = iterative_method(H0,R,VR,g,s)
>>>>>>> Stashed changes
