# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:52:12 2021

@author: Sebastian
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import pickle
import scipy
import seaborn as sns
from sklearn.decomposition import PCA
import csv
from statannot import add_stat_annotation

import scipy.spatial.distance as ssd
from Tools.analysis_tools import hierarchical_clustering, generate_cluster_map




"""
Load the dataset
"""

with open('.\\Processed data\\dark_4ss_18_light_2ss_33_log21_clean.pickle','rb') as f:
    header, seg_time, tseries, dissimilarity_matrix, labels_cond, flies = pickle.load(f)

# Dissimilarity matrix in vector-form for the Hierarchical Clustering
pdist=ssd.squareform(dissimilarity_matrix, 'precomputed') 

# Candidate windows
w = np.array(np.unique(np.round(np.logspace(1,2,20))),dtype=np.int64)[1:7]

# List of legnth of segments
seg_time_len = [x[1]-x[0] for x in seg_time]

"""
Figure 2A
"""

path_ = scipy.io.loadmat('.\\Data\\Fly1893_path_1d.mat')
path_ = path_['path']

filename = '.\\Data\\LocalCurvature' + str(1893) + '_1d_2.txt'
curvature_1d = []
with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\n')
        for row in reader:
            curvature_1d.append(float(row[0]))
            
labeled_curv_1d = ['red' if x > 1 else 'blue' for x in curvature_1d]
    

t0 = 0
seg_time_curv = []
for i in range(len(labeled_curv_1d)-1):
    if labeled_curv_1d[i] != labeled_curv_1d[i+1]:
        tf = i
        seg_time_curv.append([t0,tf])
        t0 = i
if seg_time_curv[-1][1] != len(labeled_curv_1d):
    seg_time_curv.append([seg_time_curv[-1][1]+1,len(labeled_curv_1d)])

fig2a = plt.figure(figsize=(4,4))
ax = fig2a.add_subplot(111)
for seg in seg_time_curv[:45]:
    t0, tf = seg
    ax.plot(path_[t0:tf+1,0], path_[t0:tf+1,1], color = labeled_curv_1d[tf-1])
    
    
    
"""
Figure 2B
"""

thresh = 6000
dend = hierarchical_clustering(thresh, ['red', 'black'], pdist, 'ward')
colormap = generate_cluster_map(dend)

df = pd.DataFrame([])

ax_labels = ['black', 'red']
for j, color in enumerate(ax_labels):
    amount = []
    for long in w:    
        print(color,long)
        
        amount.append(len([tseries[seg_time[i][0]:seg_time[i][1],:2] 
                                      for i, x in enumerate(zip(seg_time_len,colormap)) 
                                      if x == (long,color)]))    
    
    df = df.append(pd.DataFrame({'color':[color for x in range(len(w))],
                                 'Time (ms)':np.round(w*(100/6),1),
                                 'amount': amount}))

        
df['percentage'] = (df.groupby(['color','Time (ms)']).max()/df.groupby('color').sum()).amount.values

fig2b = plt.figure(figsize=(8,8))
ax = fig2b.add_subplot(111)

pd.pivot_table(df, index='Time (ms)', columns='color', values='percentage').plot(kind='bar', 
                                                                                 color=['black','red'],ax = ax)
ax.legend(['Non-Locomotion','Locomotion'])
ax.set_ylabel('Percentage')
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
    

"""
Figure 2C
"""
thresh = 900

# Color palette pre-computed from merging analysis
color_palette_lower = ['black', 'cyan', 'cyan', 'magenta', 'magenta', 'blue', 'blue', 'blue', 'darkviolet', 'darkviolet',
                       'chocolate', 'darkolivegreen', 'mediumspringgreen', 'mediumspringgreen', 'mediumspringgreen',
                       'dodgerblue', 'dodgerblue', 'dodgerblue', 'lime', 'lime', 'lime', 'lime', 'lime', 'salmon',
                       'salmon', 'salmon', 'salmon', 'salmon', 'salmon', 'salmon', 'salmon', 'salmon',
                       'salmon', 'salmon', 'salmon', 'salmon', 'salmon', 'salmon', 'salmon', 'salmon',
                       'salmon', 'salmon', 'salmon']

dend = hierarchical_clustering(thresh,color_palette_lower,pdist,'ward')
colormap = generate_cluster_map(dend)


"""
Figure 2D
"""

def denoise_ts(features, n_components, nComp):
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(features)
    
    mu = np.mean(features, axis=0)
    
    Xhat = np.dot(X[:,:nComp], pca.components_[:nComp,:])
    Xhat += mu
    
    return Xhat

ax_labels = ['black','cyan', 'magenta', 'blue','darkviolet','chocolate',
              'darkolivegreen','mediumspringgreen','dodgerblue','lime']

w_ = [11,16,21]
n_components = 3
nComp = 2
ts_labels = ['Rotational \n Velocity \n (°/s)', 'Forward \n Velocity \n (mm/s)']

for color in ax_labels:
    acum = 0
    acum_pc = 0
       
        
    fig2d = plt.figure(figsize=(6,6))
    ax1 = fig2d.add_subplot(411)
    ax1.set_ylim(-350,350)
    ax1.set_xlim(0,np.sum(w))
    ax1.set_ylabel('Rotational \n Velocity \n (°/s)')
    ax1.set_xticks([])
    
    ax2 = fig2d.add_subplot(412)
    ax2.set_ylim(-10,30)
    ax2.set_xlim(0,np.sum(w))
    ax2.set_ylabel('Forward \n Velocity \n (mm/s)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_xticks(np.cumsum(w))
    ax2.set_xticklabels([str(np.round(x*(100/6),2)) for x in w])
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    

    for long in w_:    
        
        timeseries = [tseries[seg_time[i][0]:seg_time[i][1],:2] 
                      for i, x in enumerate(zip(seg_time_len,colormap)) 
                      if x == (long,color)]   
        
        if timeseries:
            for ts in range(timeseries[0].data.shape[1]):
                features = pd.DataFrame(np.vstack([x[:,ts] for x in timeseries]))
                Xhat = denoise_ts(features, n_components, nComp)
    
                data = pd.DataFrame({'t': np.hstack([np.arange(acum,long+acum,1) for x in range(len(Xhat))]),
                                    ts_labels[ts]: Xhat.reshape(-1)})                   
                
                
                if ts<1:
                    sns.lineplot(x = 't', 
                                  y = ts_labels[ts], 
                                  data = data, 
                                  color = color, 
                                  ax = ax1,
                                  )
                else:
                    sns.lineplot(x = 't', 
                              y = ts_labels[ts], 
                              data = data, 
                              color = color, 
                              ax = ax2,
                              )               
        
        ax1.axvline(x = (acum+long), linestyle = '--', color = 'black', linewidth=1)
        ax2.axvline(x = (acum+long), linestyle = '--', color = 'black', linewidth=1)
        acum += long
        acum_pc += (n_components-1)
        
"""
Figure 2E - top
"""

def transition_matrix(transitions, lag):
    n = 1+ max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[lag:]):
        M[i][j] += 1

    return M

color_palette_lower = ['black', 'cyan', 'magenta', 'blue', 'darkviolet',
                       'chocolate', 'darkolivegreen', 'mediumspringgreen',
                       'dodgerblue', 'lime', 'salmon']
short_name = ['blk', 'cya', 'mag', 'blu', 'dkv', 'chc', 'dkg', 'mspg', 'dgb', 'lim', 'slm']

transitions = [i for color in colormap for i,v in enumerate(color_palette_lower) if v == color]

m = transition_matrix(transitions,1)
cm = np.reshape(np.vstack(m),(11,11))
classes = color_palette_lower
cm = pd.DataFrame(cm.copy(), index = classes)
cm.columns = classes
cm = cm.drop(index = 'salmon',columns ='salmon')
cm = cm.div(cm.sum(axis=1), axis=0)
U, s, V = np.linalg.svd(cm.copy())

#Sorted the modes
for mode in range(6):
    cm2 = s[mode]*np.outer(np.sort(U[:,mode]),np.sort(V[mode,:]))
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(111)
    ax.imshow(cm2, cmap=plt.cm.Blues)
    ax.grid('off')
    ax.tick_params(axis = 'x', rotation = 90)
    cax = ax.matshow(cm2)
    fig.colorbar(cax)
    
    tick_marks = np.arange(len(cm.index.to_numpy()))
    plt.xticks(tick_marks, cm.index.to_numpy()[np.argsort(V[mode,:])])
    plt.yticks(tick_marks, cm.index.to_numpy()[np.argsort(U[:,mode])])
    for xtick, color in zip(ax.get_xticklabels(), np.array(color_palette_lower)[np.argsort(V[mode,:])]):
        xtick.set_color(color)
    for ytick, color in zip(ax.get_yticklabels(), np.array(color_palette_lower)[np.argsort(U[:,mode])]):
        ytick.set_color(color)                
    # ax.set_ylabel('Segment [t]')     
    # ax.set_xlabel('Segment [t+1]')
        
#NOT sorted the modes
for mode in range(6):
    cm2 = s[mode]*np.outer(U[:,mode],V[mode,:])
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(111)
    ax.imshow(cm2, cmap=plt.cm.Blues)
    ax.grid('off')
    ax.tick_params(axis = 'x', rotation = 90)
    cax = ax.matshow(cm2)
    fig.colorbar(cax)
    
    tick_marks = np.arange(len(cm.index.to_numpy()))
    plt.xticks(tick_marks, cm.index.to_numpy())
    plt.yticks(tick_marks, cm.index.to_numpy())
    for xtick, color in zip(ax.get_xticklabels(), np.array(color_palette_lower)):
        xtick.set_color(color)
    for ytick, color in zip(ax.get_yticklabels(), np.array(color_palette_lower)):
        ytick.set_color(color)                



"""
Figure 2E - bottom
"""

#before/after plots average
color_t = 'cyan'
color_t1 = 'black'
win = 16
for color in [color_t]:   
    
    rot_ = []
    fwd_ = []               
    for seg_id in [i for i,v in enumerate(zip(colormap,seg_time_len)) if v == (color,win)]:
        t0, tf = seg_time[seg_id]       
        tn0, tnf = seg_time[seg_id+1] 
        
        sample_rot = np.full(60, np.nan)
        sample_fwd = np.full(60, np.nan)
        
        if colormap[seg_id+1] == color_t1: 
        
            sample_rot[:len(tseries[t0:tnf,0])] = tseries[t0:tnf,0]
            sample_fwd[:len(tseries[t0:tnf,1])] = tseries[t0:tnf,1]
            
            rot_.append(sample_rot)
            fwd_.append(sample_fwd)


fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(211)
ax1.set_ylim(-1500,1500)
ax1.set_xticklabels([])


ax2 = fig.add_subplot(212)
ax2.set_ylim(-20,50)
ax2.set_xlabel('samples')  

rot_ = np.vstack(rot_)
fwd_ = np.vstack(fwd_)
mean_rot = pd.DataFrame({'t': np.hstack([range(rot_.shape[1]) for x in range(len(rot_))]),
                                'Rotational \n Velocity \n (°/s)': rot_.reshape(-1)})
mean_fwd = pd.DataFrame({'samples': np.hstack([range(fwd_.shape[1]) for x in range(len(fwd_))]),
                                'Forward \n Velocity \n (mm/s)': fwd_.reshape(-1)})

for r, f in zip(rot_,fwd_):
    ax1.plot(np.arange(0,win+1,1),r[:win+1], color = color_t, alpha = 0.2)
    ax1.plot(np.arange(win,len(r),1),r[win:], color = color_t1, alpha = 0.2)
    ax2.plot(np.arange(0,win+1,1),f[:win+1], color = color_t, alpha = 0.2)
    ax2.plot(np.arange(win,len(r),1),f[win:], color = color_t1, alpha = 0.2)

sns.lineplot(x = 't', 
             y = 'Rotational \n Velocity \n (°/s)', 
             data = mean_rot, 
             color = 'gray', 
             ax = ax1,
             ).set_xlabel('')
sns.lineplot(x = 'samples', 
             y = 'Forward \n Velocity \n (mm/s)', 
             data = mean_fwd, 
             color = 'gray', 
             ax = ax2,
             )



"""
Figure 4G
"""

for color in ['dodgerblue', 'lime']:

    angular_dev = pd.DataFrame([])

    for lab in np.unique(labels_cond):
        for win in w:
            for seg in [seg_time[i] for i,col in enumerate(zip(colormap,labels_cond,seg_time_len)) 
                    if col == (color,lab,win)]:
                t0, tf = seg
                
                if win == 13 or win == 18:
                    if win == 13:
                        win_ = 11
                    if win == 18:
                        win_ = 21
                else:
                    win_ = win
                
                angle = np.sum(abs(tseries[t0:tf,0]))/60
                forward_dist = np.sum(abs(tseries[t0:tf,1]))/60
                angular_dev = angular_dev.append(pd.DataFrame({'Angular Deviation (°/mm)':angle/forward_dist,
                                                             'Condition':lab,
                                                             'Chunk size':win_,
                                                             'Cluster':color}, index=[0]), ignore_index=True)


    angular_dev = angular_dev[angular_dev['Angular Deviation (°/mm)']<100]
    angular_dev['Angular Deviation (°/mm)'] = np.log(angular_dev['Angular Deviation (°/mm)'])
    
    box_pairs = [
        ((11,'10d'), (11,'1d')),
        ((11,'10d'), (11,'darkness')),
        ((11,'1d'), (11,'darkness')),        
       
        ((14,'10d'), (14,'1d')),
        ((14,'10d'), (14,'darkness')),
        ((14,'1d'), (14,'darkness')),
        
        ((16,'10d'), (16,'1d')),
        ((16,'10d'), (16,'darkness')),
        ((16,'1d'), (16,'darkness')),        

        ((21,'10d'), (21,'1d')),
        ((21,'10d'), (21,'darkness')),
        ((21,'1d'), (21,'darkness')),
    
        ]
    
    fig, ax = plt.subplots(1, 1)
    sns.violinplot(x="Chunk size", 
                  y="Angular Deviation (°/mm)", 
                  hue = 'Condition',
                  data=angular_dev,
                  palette=[color,'orchid','purple'],
                  ax = ax,
                  )
   
    
    add_stat_annotation(ax, data=angular_dev, 
                        x="Chunk size", 
                        y="Angular Deviation (°/mm)", 
                        hue='Condition', 
                        box_pairs=box_pairs,
                        test='Mann-Whitney', 
                        loc='inside', verbose=2)
    ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    


                    