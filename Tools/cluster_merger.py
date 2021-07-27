# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 20:42:20 2021

@author: Sebastian
"""

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import concurrent.futures
from scipy.cluster.hierarchy import linkage
import scipy.spatial.distance as ssd
import pickle
from pairwise_comparison import compute_diss_mat
from analysis_tools import hierarchical_clustering, generate_cluster_map

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import dask_ml.model_selection as dcv


def initiliaze_params():
    """Computes the neccesary initial elements to perform the cluster merging     
    
    Parameters
    ----------

    
    Returns
    -------
    dend (dictionary): 
        Dictionary with the structures to compute the dendrogram
        
    colormap (list): 
        A list containing the color assigned to each sample according to the clustering
       performed by the dendrogram
    
    seg_time_len (list): 
        Indicates the length for each segment in seg_time
        
    w (array, int64): 
        Candidate windows in number of samples
        
    Z (array, float64): 
        Array that contains the linkage of the Hierarchical clustering
    
    color_palette (list): 
        The list containing the colors of the clusters
    
    coordinates(DataFrame):
        Coordiantes of the position of the branches of the dendrogram
        
    auc_thresh(float64):
        Threshold to decide merging a pair of clusters. Clusters are merged
        if the AUC is below the threshold --clusters can be identified as
        being different. Otherwise, they are the same.
        
    seg_time(list):
        Contains the breakpoints that define each segment [t_initial, t_end]
        
    tseries(array, float64):
        Time series with the kinematics
        
    """
    
    with open('dark_4ss_18_light_2ss_33_log21_clean.pickle','rb') as f:
        header, seg_time, tseries, dissimilarity_matrix, labels_cond, flies = pickle.load(f)
        
    pdist=ssd.squareform(dissimilarity_matrix, 'precomputed')

    thresh = 900
    color_palette_lower = [
                     'red','cyan','green', 'magenta','blue','orange', 'maroon',
                    'midnightblue','darkviolet','chocolate', 'darkolivegreen',
                    'gold', 'black', 'deeppink', 'mediumslateblue','seagreen',
                    'firebrick','sienna','sandybrown','mediumspringgreen',
                    'lightseagreen','paleturquoise','darkcyan','darkturquoise',
                    'deepskyblue','royalblue','navy','mediumpurple','darkorchid',
                    'plum', 'olivedrab','chartreuse','palevioletred','lightcoral',
                    'coral','peachpuff','navajowhite','lemonchiffon','bisque',
                    'tan','moccasin','darkkhaki','palegreen']
    
    dend = hierarchical_clustering(thresh,color_palette_lower,pdist,'ward')
    colormap = generate_cluster_map(dend)
    
    seg_time_len = [x[1]-x[0] for x in seg_time]
    w = np.array(np.unique(np.round(np.logspace(1,2,20))),dtype=np.int64)[1:7]
    Z = linkage(pdist, 'ward')
    color_palette = color_palette_lower.copy()
    decimals = 5
    coordinates = pd.DataFrame(dend['dcoord']).round(decimals)
    Z[:,2] = np.round(Z[:,2],decimals)
    auc_thresh = 0.99
    
    return dend, colormap, seg_time_len, w, Z, color_palette, coordinates, auc_thresh, seg_time, tseries

def reconstruct_original_from_PC(features, n_comp, nComp):
    """Denoises the samples by reconstructing them with fewer principal components     
    
    Parameters
    ----------
    features (float64): (n samples, p dimensions)
        Dataframe with the samples filtered in getAUC. Each row is a time series
        of a specific length
        
    n_comp (int):
        Number of components to compute PCA        
        
    nComp (int): 
       Number of components to reconstruct the samples
       nComp <= n_comp
    

    
    Returns
    -------
    Xhat (float64): (n samples, k dimensions) k<p
        Denoised samples
        
        
    """
    # scaler = StandardScaler().fit(features)
    # X = scaler.transform(features)
    X = features
    
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    scores = pca.transform(X)
    eig_vec = pca.components_
                
    # mu = scaler.mean_
    # mu = np.tile(mu,(len(features),1))
    # Xhat = np.dot(scores[:,:nComp],eig_vec[:nComp,:]) + mu  
    Xhat = np.dot(scores[:,:nComp],eig_vec[:nComp,:]) 
    
    return Xhat

def generate_dataset(seg_time_PCA, tseries_PCA, colormap, w, pair, seg_time_len):    
    """Computes a pairwise distance between the segments of the pair of cluster
        being compared. Finally, it generates the labels (y) for each row of the 
        distance matrix (X)
    
    Parameters
    ----------    
    seg_time_PCA(list):
        Contains the breakpoints that define each segment [t_initial, t_end] after
        denoising them with PCA
        
    tseries_PCA(array, float64):
        Time series with the kinematics after denoising them with PCA        
    
    colormap (list):
        A list containing the color assigned to each sample according to the clustering
       performed by the dendrogram
        
    w (array, int64):
        Candidate windows in number of samples
        
    pair (tuple): 
        Pair of clusters to be compared
        
    seg_time_len(list):
        Indicates the length for each segment in seg_time
        
    
    Returns
    -------
    X (array, float32): 
        Distance matrix generated by pairwise comparison between the segments
        of the clusters being compared        
        
    y (array, int64):
        Labels of the rows of the distance matrix. 0's for the segments in the
        cluster1 and 1's for the cluster2
        
    """
    
    cluster1, cluster2 = pair
       
    
    X = compute_diss_mat(seg_time_PCA, tseries_PCA)    
    
    y = pd.DataFrame([])
    for i, color in enumerate([cluster1, cluster2]):    
        label = []
        for long in w:
            for seg_id in [i for i,v in enumerate(zip(colormap,seg_time_len)) if v == (color,long)]:
                label.append(i)                
        y = y.append(pd.DataFrame({'Label':label}), ignore_index=True)

    y = y['Label'].to_numpy()
    
    return X, y
    

def getAUC(pair, colormap, seg_time, tseries):    
    """Computes the Area under the ROC curve for the binary classification
        between samples of neighbouring clusters    
    
    Parameters
    ----------
    pair (tuple): 
        
        
    colormap (list): 
        
        
           
    
    Returns
    -------
    score (float): 
        
        
    """
    
    cluster1, cluster2 = pair   
    seg_time_len = [x[1]-x[0] for x in seg_time]
    
    #Reconstructed PCA segments to compare with DTW
    tseries_PCA = np.zeros((4,1))
    seg_time_PCA = [[[0,0]]]
    initial = True
    for color in [cluster1, cluster2]:
        nComp = 2
        n_comp = 3
        
        for long in w:    
            timeseries = [tseries[seg_time[i][0]:seg_time[i][1],:] 
                          for i, x in enumerate(zip(seg_time_len,colormap)) 
                          if x == (long,color)]   
            
            if timeseries:
                features = pd.DataFrame(np.vstack([x[:,0] for x in timeseries]))   
                time_points = len(features)*long
                Xhat = np.zeros((4,time_points))
                
                for i in range(len(Xhat)):
                    features = pd.DataFrame(np.vstack([x[:,i] for x in timeseries]))
                    if len(features)>=3:
                        Xhat[i,:] = reconstruct_original_from_PC(features, n_comp, nComp).reshape(1,time_points)
                    else:
                        Xhat[i,:] = features.to_numpy().reshape(1, time_points)
                
            
                
                tseries_PCA = np.hstack((tseries_PCA,Xhat))
                
                indexes = [int(x) for x in np.linspace(0,time_points,len(features)+1)]
                indexes = np.vstack(([x for x in indexes],np.roll([x-1 for x in indexes],-1)))[:,:-1].T
                if initial:
                    last = seg_time_PCA[-1][-1][1]                
                else:
                    last = seg_time_PCA[-1][-1][1] + 1
                initial = False
                
                seg_time_PCA.append([[x[0]+last,x[1]+last] for x in indexes])
    
    tseries_PCA = tseries_PCA[:,1:].T
    seg_time_PCA = np.vstack(seg_time_PCA)[1:,:]
    seg_time_PCA = [[x[0],x[1]] for x in seg_time_PCA]
    
    X, y = generate_dataset(seg_time_PCA, tseries_PCA, colormap, w, pair, seg_time_len)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    parameters = {'C':np.logspace(-10, 3, 20)}
    lr = LogisticRegression(random_state=34,
                            solver = 'liblinear',
                            max_iter = 1000,
                            verbose = 1)
    model_ = lr
    
       
    clf = dcv.RandomizedSearchCV(model_, 
                       parameters, 
                       scoring = 'roc_auc',
                       cv=10, 
                       return_train_score=True)
    clf.fit(X_train.copy(), y_train.copy())
    print(clf.best_estimator_)
    
    return clf.score(X_test,y_test)
    



#Merging clusters            
            
def get_neighbor_pairs(dend, color_palette, Z):    
    """Computes the pair of clusters that share the same common branch. It returns 
        all the pairs and their individual distance to the common branch
    
    Parameters
    ----------
    dend (dictionary): 
        Dictionary with the structures to compute the dendrogram        
        
    color_palette (list): 
        The list containing the colors of the clusters
        
    Z (array, float64): 
        Array that contains the linkage of the Hierarchical clustering
        
    
    Returns
    -------
    pairs (array): 
        Array containing the pairs of clusters that share the same common branch,
        here referred as to neighbors
        
    table_clust_dist(Dataframe):
        Dataframe summarizing the distance of each pair of clusters to the
        common branch
        
        
    """
    
    decimals = 5 #Decimals to approximate the coordinates
    max_clust_height = pd.DataFrame({'cluster':dend['color_list'], 
                                     'dcoord':np.max(dend['dcoord'], axis = 1)}) \
                                            .groupby(['cluster'], sort=False)['dcoord'] \
                                            .max() \
                                            .round(decimals)
                            
    num_clusters = len(np.unique(color_palette))
    cluster_dist_prox = np.zeros((1,num_clusters)).T
    
    for i, color in enumerate(np.unique(color_palette)):
        idx = np.where(Z[:,2]==max_clust_height.loc[color])[0][0] + 1 + len(Z)
        cluster_dist_prox[i,:] = Z[np.where(Z == idx)[0][0],2]
    table_clust_dist = pd.DataFrame(cluster_dist_prox, 
                                    index = np.unique(color_palette))
    
    pairs = table_clust_dist[table_clust_dist.duplicated(keep=False)] \
                            .sort_values(by=0) \
                            .index \
                            .to_numpy() \
                            .reshape(-1,2)
    return pairs, table_clust_dist


def check_pairs(pairs, previous_pairs):    
    """Checks 
    
    Parameters
    ----------
    pairs (array): 
        Current neighboring pairs of clusters to be assessed
        
    previous_pairs (array): 
        Previous pairs of clusters already processed
              
    
    Returns
    -------
    valid_pairs (list): 
        List of pairs to be assessed in the next iteration         
        
    """
    
    valid_pairs = []
    for pair in pairs:
        p1, p2 = pair
        pair = [p1,p2]

        for i, pp  in enumerate(previous_pairs):
            pp1, pp2 = pp
            pp = [pp1,pp2]

            
            if (pp==pair) or (pp[::-1]==pair) or (pp==pair[::-1]) or (pp[::-1]==pair[::-1]):
                break
            if i == len(previous_pairs)-1:
                valid_pairs.append(pair)
    return valid_pairs



if __name__ == "__main__":
    
    dend, colormap, seg_time_len, w, Z, color_palette, coordinates, auc_thresh, seg_time, tseries = initiliaze_params()
    
    pairs, table_clust_dist = get_neighbor_pairs(dend, color_palette, Z)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        future_pair = {executor.submit(getAUC, pair, colormap, seg_time, tseries): pair for pair in pairs}
        for future in concurrent.futures.as_completed(future_pair):
            pair = future_pair[future]
            print(pair)
            try:
                result = future.result()
                if result < auc_thresh:      
                    print('Merge them!')
                    clust1, clust2 = pair
                    color_palette = [clust1 if x == clust2 else x for x in color_palette]
                    colormap = [clust1 if x == clust2 else x for x in colormap]
                    max_height = table_clust_dist.loc[clust1][0]
                    color_idx = np.where(coordinates.iloc[:,1:3]==np.ones((1,2))*max_height)[0][0]
                    
                    dend['color_list'][color_idx] = clust1 
    
        
            except Exception as exc:
                print('%r generated an exception: %s' % (pair, exc))
            
    previous_pairs = pairs.copy()
    print('--------------')
    
    while True:
        pairs, table_clust_dist = get_neighbor_pairs(dend, color_palette, Z)
        pairs = check_pairs(pairs, previous_pairs)
        if not pairs:
            break
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            future_pair = {executor.submit(getAUC, pair, colormap, seg_time, tseries): pair for pair in pairs}
            for future in concurrent.futures.as_completed(future_pair):
                pair = future_pair[future]
                print(pair)
                try:
                    result = future.result()
                    if result < auc_thresh:      
                        print('Merge them!')
                        clust1, clust2 = pair
                        color_palette = [clust1 if x == clust2 else x for x in color_palette]
                        colormap = [clust1 if x == clust2 else x for x in colormap]
                        max_height = table_clust_dist.loc[clust1][0]
                        color_idx = np.where(coordinates.iloc[:,1:3]==np.ones((1,2))*max_height)[0][0]
                        
                        dend['color_list'][color_idx] = clust1 
        
            
                except Exception as exc:
                    print('%r generated an exception: %s' % (pair, exc))
                
        previous_pairs = np.vstack((pairs, previous_pairs))
        print('--------------')
        
    
                    
