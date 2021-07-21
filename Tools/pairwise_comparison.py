# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:57:42 2021

@author: Sebastian
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io
import itertools
from itertools import chain, repeat, combinations, islice
import xarray as xr
import pandas as pd
import pickle
import numpy.ma as ma
import math
from joblib import Parallel, delayed
from numba import jit


@jit
def myGetTheta_m(y,l1):
    
    yp = np.concatenate((y[:l1-1,:],y[l1:-1,:]))
    yf = np.concatenate((y[1:l1,:],y[l1+1:,:]))
    pred = np.full((y.shape[0],y.shape[1]),np.nan)
    pos = np.full((y.shape[0],1),True).reshape(-1)
    pos[np.array([0,l1])] = False
    
        
    x_inter = np.concatenate((np.ones((yp.shape[0],1)),yp),axis=1)
    pinverse = np.linalg.pinv(x_inter)
    beta = np.dot(pinverse,yf)
    pred[pos,:] = np.dot(x_inter,beta)
    eps=y-pred
    cov=np.cov(eps[~np.isnan(eps[:,0]),:].T)
    theta=np.concatenate((beta,cov))
    
    return theta


@jit
def myLogLik_m(theta, y):
    T = y.shape[0]
    dim = y.shape[1]
    clag = 1
    inter = theta[0] # mean
    coef = theta[1:clag*theta.shape[1]+1] #coupling matrix
    sigma = theta[clag*theta.shape[1]+1:] # covariance matrix
#    yf = y[clag:]
    yp = y[:-clag]
    
    sigmainv=np.linalg.pinv(sigma)
    abs_log_det_inv_s = np.linalg.slogdet(sigmainv)[1]
    
    x_inter2=np.concatenate((np.ones((yp.shape[0],1)),yp),axis=1)
    beta2=np.concatenate((inter.T.reshape(1,-1),coef))
    
    pred2 = np.full((y.shape[0],y.shape[1]),np.nan)
    pos = np.full((y.shape[0],1),True).reshape(-1)
    pos[np.array([0])] = False
    pred2[pos,:]=np.dot(x_inter2,beta2)
    eps2=y-pred2
    eps2 = eps2[~np.isnan(eps2[:,0]),:]
    
    elem_sum = np.ones((1,len(eps2)))
    for i in range(len(eps2)):
        elem_sum[:,i] = np.dot(eps2[i].T, np.dot(sigmainv,eps2[i]))
    last_term2 = np.sum(elem_sum)
    
    result = (-0.5*dim*(T-1)*math.log(2*math.pi) + 0.5*(T-1)*abs_log_det_inv_s
            -0.5*last_term2)
   
    return result

def compute_diss_mat(seg_time, tseries):
    
    c = combinations(range(len(seg_time)),2)
    dissimilarity_matrix = np.zeros((len(seg_time),len(seg_time)))
    dissimilarity_matrix = np.float32(dissimilarity_matrix)
    tseries = np.float64(tseries)
    
    for model in tqdm(list(c)):
        modelx, modely = model 
        t0_w1, tf_w1 = seg_time[modelx]
        t0_w2, tf_w2 = seg_time[modely]
        ts1 = tseries[t0_w1:tf_w1]
        ts2 = tseries[t0_w2:tf_w2]
        
        master_theta = myGetTheta_m(np.concatenate((ts1, ts2)), ts1.shape[0])
        
        loglik_ac = myLogLik_m(myGetTheta_m(ts1,ts1.shape[0]),ts1) - myLogLik_m(master_theta,ts1)
        loglik_bc = myLogLik_m(myGetTheta_m(ts2,ts2.shape[0]),ts2) - myLogLik_m(master_theta,ts2)
        
        distance_ab = np.sum(loglik_ac + loglik_bc) # likelihood_distance  
        dissimilarity_matrix[modelx, modely] = np.float32(distance_ab)
    
    dissimilarity_matrix = dissimilarity_matrix + dissimilarity_matrix.T
    
    return dissimilarity_matrix


if __name__ == "__main__":
    
    from analysis_tools import concaTseries
    
    flies = list(np.concatenate((np.arange(1641,1645,1),np.arange(1646,1654,1),np.arange(1655,1659,1),[1660,1661])))
    filename = ['Fly_','_dark_hang_975.pickle']
    seg_time_dark, tseries_dark = concaTseries(flies,filename)
    
    flies = list(np.concatenate(([1893,1894],np.arange(1896,1912,1),np.arange(2101,2113,1),np.arange(2114,2117,1))))
    filename = ['Fly_','_1d_hang_975.pickle']
    seg_time_1d, tseries_1d = concaTseries(flies,filename)
    
    filename = ['Fly_','_10d_hang_975.pickle']
    seg_time_10d, tseries_10d = concaTseries(flies,filename)
    
    seg_time_10d = [[x[0] + seg_time_1d[-1][1], x[1] + seg_time_1d[-1][1]] for x in seg_time_10d]
    seg_time_dark = [[x[0] + seg_time_10d[-1][1], x[1] + seg_time_10d[-1][1]] for x in seg_time_dark]
    
    seg_time = list(itertools.chain(seg_time_1d,seg_time_10d, seg_time_dark))
    tseries = np.concatenate((tseries_1d,tseries_10d,tseries_dark))
    
    dissimilarity_matrix = compute_diss_mat(seg_time, tseries)
    
    
    