# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 20:42:20 2021

@author: Sebastian
"""

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.cluster.hierarchy import linkage
from dissimilarity_matrix.pairwise_comparison import compute_diss_mat

def reconstruct_original_from_PC(features, n_comp, nComp):
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

def getAUC(pair, colormap):
    cluster1, cluster2 = pair   

    #Reconstructed PCA segments 
    tseries_PCA = np.zeros((4,1))
    seg_time_PCA = [[[0,0]]]
    initial = True
    for color in [cluster1, cluster2]:
        nComp = 2
        
        for long in w:    
            
            timeseries = [tseries[seg_time[i][0]:seg_time[i][1],:] 
                          for i, x in enumerate(zip(seg_time_len,colormap)) 
                          if x == (long,color)]   
            
            if timeseries:
                features = pd.DataFrame(np.vstack([x[:,0] for x in timeseries]))   
                time_points = len(features)*long
                Xhat = np.zeros((4,time_points))
                Xhat[0,:] = reconstruct_original_from_PC(features, 2, nComp).reshape(1,time_points)
                
                for i in range(1,timeseries[0].shape[1]):
                    features = pd.DataFrame(np.vstack([x[:,i] for x in timeseries]))
                    Xhat[i,:] = reconstruct_original_from_PC(features, 2, nComp).reshape(1,time_points)
                
                
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
    # tseries_PCA = tseries_PCA[:,0]
    seg_time_PCA = np.vstack(seg_time_PCA)[1:,:]
    seg_time_PCA = [[x[0],x[1]] for x in seg_time_PCA]
    
    samples = []
    for color in [cluster1,cluster2]:    
        for long in w:
            for seg_id in [i for i,v in enumerate(zip(colormap,seg_time_len)) if v == (color,long)]:
                samples.append(seg_id)
    
    c = list(combinations(range(len(samples)),2))
    pw_mat = np.zeros((len(samples),len(samples)))
    
    for model in c:
        modelx, modely = model 

        t0_w1, tf_w1 = seg_time_PCA[modelx]
        t0_w2, tf_w2 = seg_time_PCA[modely]
        ts1 = tseries_PCA[t0_w1:tf_w1]
        ts2 = tseries_PCA[t0_w2:tf_w2]
        
        master_theta = myGetTheta_m(np.concatenate((ts1, ts2)), ts1.shape[0])
        
        loglik_ac = myLogLik_m(myGetTheta_m(ts1,ts1.shape[0]),ts1) - myLogLik_m(master_theta,ts1)
        loglik_bc = myLogLik_m(myGetTheta_m(ts2,ts2.shape[0]),ts2) - myLogLik_m(master_theta,ts2)
        
        distance_ab = np.sum(loglik_ac + loglik_bc) 
        pw_mat[modelx, modely] = np.float32(distance_ab)
    pw_mat = pw_mat + pw_mat.T

    
    samples1 = []
    for color in [cluster1]:    
        for long in w:
            for seg_id in [i for i,v in enumerate(zip(colormap,seg_time_len)) if v == (color,long)]:
                samples1.append(seg_id)     

    
    samples2 = []
    for color in [cluster2]:    
        for long in w:
            for seg_id in [i for i,v in enumerate(zip(colormap,seg_time_len)) if v == (color,long)]:
                samples2.append(seg_id) 
    

    y = np.concatenate((np.ones((len(samples1))),np.zeros((len(samples2)))))
    
    X_train, X_test, y_train, y_test = train_test_split(pw_mat, y, test_size=0.3, random_state=42)
    
    parameters = {'C':np.logspace(-10, 3, 20)}
    lr = LogisticRegression(random_state=34,
                            solver = 'liblinear',
                            max_iter = 1000,
                            verbose = 1)
    model_ = lr
    
    # parameters = {'n_estimators':[200,400,600,800,1000],
    #               'max_depth':[2, 6, 10, 14, 18]}
    # rf = RandomForestClassifier(random_state=0, criterion = 'entropy')
    # model_ = rf
    
    clf = dcv.RandomizedSearchCV(model_, 
                        parameters, 
                        scoring = 'roc_auc',
                        cv=5, 
                        return_train_score=True)
    clf.fit(X_train.copy(), y_train.copy())
    print(clf.best_estimator_)
    
    return clf.score(X_test,y_test)
    



#Merging clusters            
            
def get_neighbor_pairs(dend, color_palette, Z):
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

    seg_time_len = [x[1]-x[0] for x in seg_time]
    w = np.unique(seg_time_len)
    Z = linkage(pdist, 'ward')
    color_palette = color_palette_lower.copy()
    decimals = 5
    coordinates = pd.DataFrame(dend['dcoord']).round(decimals)
    Z[:,2] = np.round(Z[:,2],decimals)
    auc_thresh = 0.99
    
    pairs, table_clust_dist = get_neighbor_pairs(dend, color_palette, Z)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        future_pair = {executor.submit(getAUC, pair, colormap): pair for pair in pairs}
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
            future_pair = {executor.submit(getAUC, pair, colormap): pair for pair in pairs}
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
        
    
                    
        return colormap