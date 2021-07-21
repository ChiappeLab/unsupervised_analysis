# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:01:52 2021

@author: Sebastian
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib as mpl
import h5py
import numpy.ma as ma
import math
import pandas as pd
from tqdm import tqdm
import scipy.io
from scipy import stats
from scipy.signal import savgol_filter
from collections import defaultdict
import timeit
import pickle
from itertools import combinations, permutations, product, chain, repeat
import scipy
import pylab
from tqdm import tqdm
import seaborn as sns
import os.path
from numba import jit
import itertools
from itertools import combinations, islice
from tqdm import tqdm

#load clustering packages
import scipy.cluster as cl
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform

from pairwise_comparison import compute_diss_mat


#Parallel processing
# from sklearn.externals.joblib import Parallel, delayed
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def hierarchical_clustering(threshold, color_palette, input_matrix, method_):
        pdist = input_matrix
        thresh = threshold
        
        if color_palette:
            shc.set_link_color_palette(color_palette)
        else:
            cmap = cm.rainbow(np.linspace(0, 1, len(np.unique(shc.cut_tree(shc.linkage(pdist, method = method_),
                                                                           height=thresh)))))
            shc.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
            
        plt.figure(figsize=(15,7))
        
        dend = shc.dendrogram(shc.linkage(pdist, 
                                          method=method_),
                              color_threshold=thresh,  above_threshold_color='gray')
        plt.axhline(y = thresh, color = 'grey', linestyle = '--')
        plt.title('Dendrogram threshold = ' + str(thresh))
        plt.tick_params(labelbottom=False)
        
        return dend    
        
            
def generate_cluster_map(dend):
    """Find all the bouts in the experiment and saves the result in a csv file.

        Parameters
        ----------
        frame_rate : float, optional
            The frame rate at which the data were acquired. Default is 500 fps.

        threshold : float, optional
            The threshold to use for defining bouts.

        min_length : float, optional
            The minimum time window (in seconds) over which frames above threshold are considered a bout.

        check_ROI : bool, optional
            Check whether the tail falls within the ROI over the entire duration of each bout. Default is True.

        Returns
        -------
        bouts_df : pd.DataFrame
            DataFrame containing every bout in the experiment. Provides the fish ID, video code, and first and last
            frames of each bout. An optional column states whether the bout falls within the ROI.

        Notes
        --------
        - Values for threshold and min_length can be optimised using the 'set_bout_detection_thresholds' method.
        - This method will simply open and return the bouts DataFrame if all videos have been checked for bouts.
        """
    
    cluster_idxs = defaultdict(list)
    for c, pi in zip(dend['color_list'], dend['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(dend['ivl'][int(i)])
    
    
    #Extract colors of dendrogram leafes            
    colormap = []
    for num in range(len(dend['leaves'])):
        for i in cluster_idxs.items():
            if str(num) in i[1]:
                colormap.append(i[0])
    
    return colormap
                
def concaTseries(flies,filename):
    """Concatenates separately chunked time series and sums up the equivalent samples to the segments    
    
    Parameters
    ----------
    flies (int): 
        The list containing the labels of the flies to be concatenated
    filename (string): 
        The list containing in the first position the initial string of the 
        file to be opened and in the second position the string after the 
        label and the extension of the file
    
    Returns
    -------
    tseries (float64): 
        Concatenated time series
    seg_time (int): 
        Concatenated and updated segments
    """
    
    for i,v in enumerate(flies):
        if i == 0:            
            with open(filename[0] + str(v) + filename[1], 'rb') as f:
                seg_time0, tseries0  = pickle.load(f)
            seg_time = seg_time0
            tseries = tseries0
        else:
            with open(filename[0] + str(v) + filename[1], 'rb') as f:
                seg_time1, tseries1  = pickle.load(f)
            seg_time1 = [[x[0] + seg_time0[-1][1], x[1] + seg_time0[-1][1]] for x in seg_time1]

            seg_time = list(itertools.chain(seg_time, seg_time1))
            tseries = np.concatenate((tseries,tseries1))
            seg_time0 = seg_time1
    return seg_time, tseries
                

