# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:01:52 2021

@author: Sebastian
"""

# import os
# os.chdir('.\\Tools\\')

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.pyplot import cm
from itertools import chain
import matplotlib as mpl
import pickle
import scipy.cluster.hierarchy as shc



def hierarchical_clustering(threshold, color_palette, input_matrix, method_):
    """Performs Hierarchical clustering on the input matrix and generates clusters
        under the set threshold    
    
    Parameters
    ----------
    threshold (int): 
        Indicates the threshold to cut off the dendrogram. It colors the branches
        under every node that intercetps the threshold
        
    color_palette (list): 
        The list containing the colors for each of the clusters
        
    input_matrix (float64): 
        Matrix in squareform to perform the Hierarchical clustering 
    
    method_ (string): 
        Linkage criterion to group the samples. 
        from sklearn documentation: {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
        
    
    Returns
    -------
    dend (dictionary): 
        Dictionary with the structures to compute the dendrogram
        
    """
    
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
    """Maps the clusters' colors to each sample of the dendrogram    
    
    Parameters
    ----------
    dend (dictionary): 
        Dictionary with the structures to compute the dendrogram.
        It can be computed using the function hierarchical_clustering()     
    
    Returns
    -------
    colormap (list): 
       A list containing the color assigned to each sample according to the clustering
       performed by the dendrogram
        
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

            seg_time = list(chain(seg_time, seg_time1))
            tseries = np.concatenate((tseries,tseries1))
            seg_time0 = seg_time1
    return seg_time, tseries
                

