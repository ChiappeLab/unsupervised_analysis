# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:49:05 2021

@author: Sebastian
"""

#data format library
import h5py
#numpy
import numpy as np
import numpy.ma as ma
import os
import math
import scipy.io
from scipy.signal import savgol_filter

#Parallel processing
from joblib import Parallel, delayed
import multiprocessing


def decomposed_theta(theta, lag):
    '''
    Decomposed theta into the mean, inter, coupling matrix, coef, and covariance matrix of the error, cov
    '''
    clag = lag
    dim = theta.shape[1]
    inter = theta[0]
    coef = theta[1:clag*dim+1]
    cov = theta[clag*dim+1:]
    return inter, coef, cov

def get_yp(y, lag):
    clag = lag
    if clag<2:
        yp = y[0:-clag]
    else:
        ypredictor=[y[clag-p:-p] for p in range(1,clag+1)]
        yp = np.hstack(ypredictor)
    return yp

def get_theta(y, lag):
    '''
    Fits a AR(lag) model to y
    '''
    clag = lag
    yf = y[clag:]
    yp = get_yp(y,clag)
    y_len = y.shape[0]
    y_dim = y.shape[1]

    yp_len = yp.shape[0]
    yp_dim = yp.shape[1]
    x_inter = np.vstack((np.ones(yp_len),yp.T)).T
    pinverse = np.linalg.pinv(x_inter)
    beta = np.dot(pinverse,yf)
    pred = np.dot(x_inter,beta)
    eps = ma.zeros((y_len,y_dim))
    eps[clag:] = yf-pred
    eps[:clag] = ma.masked
    cov = np.cov(eps[clag:].T)
    theta = np.vstack((beta,cov))

    return theta, eps


def loglik_mvn(theta, y, lag):
    '''
    Computes the loglikelihood of the parameters theta fitting y
    '''

    T = y.shape[0]
    dim = y.shape[1]
    clag = lag

    inter,coef,sigma=decomposed_theta(theta,clag)
    y_len = y.shape[0]
    y_dim = y.shape[1]

    yf = y[clag:]
    yp = get_yp(y,clag)

    yp_len = yp.shape[0]

    sigmainv=np.linalg.pinv(sigma)
    sign,value = np.linalg.slogdet(sigmainv)
    abs_log_det_inv_s = value
    # cdef double abs_log_det_inv_s = np.abs(det_inv_s)

    beta = np.vstack((inter,coef))
    x_inter = np.vstack((np.ones(yp_len),yp.T)).T
    pred = np.dot(x_inter,beta)
    eps = yf-pred
    eps_len = eps.shape[0]
    element_sum = np.zeros(eps_len,dtype=np.float64)
    unmasked_time = np.arange(0,eps_len,dtype=np.dtype("i"))
    last_term = 0
    for t in unmasked_time:
        eps_t=eps[t]
        last_term+=np.dot(eps_t.T,np.dot(sigmainv,eps_t))


    return (-0.5*dim*(T-1)*math.log(2*math.pi)
            +0.5*(T-1)*abs_log_det_inv_s
            -0.5*last_term)

def gen_obs(theta, y, lag):
    '''
    Returns a simulation of an AR process
    theta are the parameters of the model
    y is the interval over which we want to simulate
    '''
    clag = lag
    dim = y.shape[1]
    size_y = y.shape[0]
    sim = np.zeros((size_y,dim),dtype=np.float64)
    #decompose theta
    inter,coef,cov=decomposed_theta(theta,clag)
    #draw an error vector with the given covariance and mean 0
    eps = np.random.multivariate_normal(np.zeros(dim),cov,size_y)
    #define the initial condition for the simulation
    sim[:clag]=y[:clag]
    for i in range(clag,len(sim)):
        sim[i,:]=inter+np.sum([np.dot(sim[i-p-1,:],coef[p*dim:(p+1)*dim]) for p in range(clag)],axis=0)+eps[i]
    return sim


def R_null(theta, window1, window2, N, lag):
    '''
    Returns the likelihood distribution
    theta is the null model
    '''

    cN = N
    clag = lag
    r_surr = []
    
    def processInput(i):
        #simulate a time series of the size of window2 using the null model, theta
        obs=gen_obs(theta,window2,clag)
        #fit a linear model to a window of size window1 and window2 in the simulation
        theta_g1,eps_g1=get_theta(obs[:len(window1)],clag)
        theta_g2,eps_g2=get_theta(obs,clag)
        #obtain the respective likelihood ratios
        loglik1=loglik_mvn(theta_g1,obs, lag)
        loglik2=loglik_mvn(theta_g2,obs, lag)
        r=loglik2-loglik1
        
        return r
    
    num_cores = multiprocessing.cpu_count()    
    r_surr.append(Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in range(cN)))

    return r_surr

def test(window1,window2,theta_1,theta_2,N,per,lag):
    #Assume theta_1 is the null model fitting window2
    r_1 = loglik_mvn(theta_2,window2, lag) - loglik_mvn(theta_1,window2, lag) #lik ratio between theta_2 and theta_1 modelling window2
    Rnull_1 = R_null(theta_1,window1,window2,N,lag) #lik ratio dist with theta_1 as the null model
    thresh_1_max=np.nanpercentile(Rnull_1,per) #threshold lik ratio from per-th percentile of null distribution
    is_in_1=r_1<thresh_1_max #check whether r_1 falls inside the distribution
    if is_in_1:
        return False
    else:
        return True

def breakfinder(ts,br,w,N,per,lag,cond_thresh):
    '''
    Look around each break to identify weather it was real or artificial
    ts is the time series we which to segment
    br is the set of breaks found using r_window
    w_step is a dictionary containing steps and the respective candidate windows
    defined for the artifical break finder
    N in the number of observations in the null likelihood ratio
    per is the percentile of the null likelihood ratio distribution used as a threshold
    '''
    steps=np.unique(np.diff(w))
    w_step={}
    w_step[steps[0]]=w[:np.arange(len(w)-1)[np.diff(w)==steps[0]][-1]+2]
    for i in range(len(steps)-1):
        w_step[steps[i+1]]=w[np.arange(len(w)-1)[np.diff(w)==steps[i]][-1]+1:np.arange(len(w)-1)[np.diff(w)==steps[i+1]][-1]+2]
    for step in w_step.keys():
        br_w=w_step[step]
        min_w=np.min(br_w)
        start=br-min_w
        #this step is probably not needed since br>=min(window_sizes)
        if start<0:
            start=0
        for i in range(len(br_w)-1) :
            w1=ts[start:br]
            w2=ts[start:br+step]
            if ma.count_masked(w1,axis=0)[0]<ma.count_masked(w2,axis=0)[0]:
                return br
            else:
                w1=np.array(w1)
                w2=np.array(w2)
                theta_1,eps1= get_theta(w1,lag)
                theta_2,eps2= get_theta(w2,lag)
                c1,A1,cov1= decomposed_theta(theta_1, lag)
                c2,A2,cov2= decomposed_theta(theta_2, lag)
                cond1=np.linalg.cond(cov1)
                cond2=np.linalg.cond(cov2)
                if cond1>cond_thresh or cond2>cond_thresh:
                    continue
                else:
                    first_test=test(w1,w2,theta_1,theta_2,N,per,lag)
                    if first_test:
                        return br
                    else:
                        continue
            start=br-br_w[i+1]
    return False


def segment_maskedArray(tseries,min_size=50):
    '''
    Segments  time series in case it has missing data
    '''
    segments=[]
    t0=0
    tf=1
    while tf<len(tseries):
        #if the first element is masked, increase it
        if np.any(tseries[t0].mask)==True:
            t0+=1
            tf=t0+1
        #if not, check tf
        else:
            #if tf is masked, the segment goes till tf
            if np.any(tseries[tf].mask)==True:
                segments.append([t0,tf])
                t0=tf+1
                tf=t0+1
            #if not increase tf
            else:
                tf+=1
    segments.append([t0,len(tseries)])
    #segments with less than min_size frames are excluded
    i=0
    while i<len(segments):
        t0,tf=segments[i]
        if tf-t0<min_size:
            segments.pop(i)
        else:
            i+=1
    return segments

def r_window(tseries,t,windows,N,per,lag,cond_thresh):
    '''
    Returns the break found after t, iterating over a set of candidate windows
    tseries is the time series w5 want to segment
    windows is the set of candidate windows
    N in the number of observations in the null likelihood ratio
    per is the percentile of the null likelihood ratio distribution used as a threshold
    '''
    for i in range(len(windows)-1):
        window1=np.array(tseries[t:t+windows[i]])
        window2=np.array(tseries[t:t+windows[i+1]])
        #fit theta_1 and theta_2 to window1 and window2
        theta_1,eps1 = get_theta(window1,lag)
        theta_2,eps2 = get_theta(window2,lag)
        #decompose theta into its components
        c1,A1,cov1 = decomposed_theta(theta_1, lag)
        c2,A2,cov2 = decomposed_theta(theta_2, lag)
        #compute the condition number of the covariance
        cond1=np.linalg.cond(cov1)
        cond2=np.linalg.cond(cov2)
        #if the condition number of the covariance matrices is above cond_thresh, increase window size
        if cond1>cond_thresh or cond2>cond_thresh:
            print('conditional number')
            continue
        else:
            #test whether there's a break between window1 and window2
            if test(window1,window2,theta_1,theta_2,N,per,lag):
                return windows[i]
            else:
                continue
    #if no break is found, return the last candidate window
    return windows[i+1]

def change_point(w,N,per,tseries,min_size,lag,cond_thresh):
    '''
    Segments an entire time series
    Returns the breaks found in each of the non masked segments, as well as the non masked segments
    tseries is the time series we which to segment
    w is the set of candidate windows
    N in the number of observations in the null likelihood ratio
    per is the percentile of the null likelihood ratio distribution used as a threshold
    '''
    #segment the array into nonmasked segments with at least min_size observations
    #segments=segment_maskedArray(tseries,min_size=min_size)
    segments=segment_maskedArray(tseries,min_size=min_size)
    windows_segment=[]
    for segment in segments:
        t0,tf=segment
        #define a new time series to be segmented
        ts=tseries[t0:tf]
        t=0
        windows=[]
        w_seg=np.copy(w)
        while t<len(ts)-w_seg[0]:
            while t>len(ts)-w_seg[-1]:
                w_seg=np.delete(w_seg,-1)
            if len(w_seg)<2:
                windows.append([t+t0,t0+len(ts)])
                break
            #estimate the first break in ts after t
            k=r_window(ts,t,w_seg,N,per,lag,cond_thresh)
            if k!=False:
                t+=k
                #If the t+min_step is larger than the len of the time series set it
                #to the end of the segment
                if len(ts)-t<=w_seg[0]:
                    #the last interval size is simply the distance from the previous break to
                    #the end of the time series
                    windows.append([t0+t-k,t0+len(ts)])
                    print('Break: ', k, 'Time: ',t, 'log 1')
                else:
                    windows.append([t+t0-k,t+t0])
                    print('Break: ', k, 'Time: ',t, 'log 2')
            else:
                t+=w_seg[0]
                print('Break: ', k, 'Time: ',t, 'log 3')
        #RemoveArtificialBreaks
        nwindows=list(windows)
        max_intervals=[]
        for i in range(len(windows)-1):
            if np.diff(windows[i])==max(w):
                max_intervals.append([i,windows[i]])
        i=0
        while i < len(max_intervals):
            k,interval=max_intervals[i]
            is_it=breakfinder(tseries,interval[-1],w,N,per,lag,cond_thresh) #last element might be articial
            if is_it!=False: #real break is found
                max_intervals.pop(i)
            else:
                nwindows[k+1][0]=nwindows[k][0]
                nwindows.pop(k)
                if len(max_intervals)>1:
                    for j in range(len(max_intervals[i:])-1):
                        max_intervals[i+j+1][0]=max_intervals[i+j+1][0]-1
                max_intervals.pop(i)
        windows_segment.append(windows)
    return windows_segment,segments


def openData(filename):    
    
    tseries = scipy.io.loadmat(filename)
    tseries = tseries['data']
    rot = tseries[:,0]
    forwd = tseries[:,1]
   
    
    fs = 60
    ts = 1/fs
    
    rot_acc = savgol_filter(np.diff(rot)/ts, 31, 3)
    rot_acc = np.append(rot_acc,rot_acc[-1])
 
    
    forwd_acc = savgol_filter(np.diff(forwd)/ts, 31, 3)
    forwd_acc = np.append(forwd_acc,forwd_acc[-1])
   
    tseries = np.concatenate((
                            rot.reshape(-1,1),
                            forwd.reshape(-1,1),
                            rot_acc.reshape(-1,1),
                            forwd_acc.reshape(-1,1)
                            ),axis = 1)

            
    ts_masked = ma.masked_array(tseries, mask=np.zeros(tseries.shape,dtype=bool))


    return ts_masked



if __name__ == "__main__":
    
    N=1000
    per=98.5
    w = np.array(np.unique(np.round(np.logspace(1,2,20))),dtype=np.int64)[1:7]
    lag = 1
    
    cwd = os.getcwd()
    filename = 'Fly_2112_1d.mat'
    tseries = openData('C:\\Users\\Sebastian\\Documents\\Paper Tomas\\Data\\' + filename)
    breaks_segments = change_point(w,N,per,tseries,20, lag = 1, cond_thresh=1e6)
    
    