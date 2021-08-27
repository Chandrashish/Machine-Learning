#
__author__ = "Chandrashish Prasad"
__license__ = "Feel free to copy, I appreciate if you learn from here"

#READING

import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import time

import Jaccard_K_clusters.py

start = time.time()
nips_kmeans = jaccard_kmeans("docword.nips.txt", vocab = "vocab.nips.txt")
end = time.time()
print(end-start," secs is the time spent on doc creation")



#Training with random seed initialisation ------------------------------------------------

inertia_nips = [0]
for i in range(1,51):
    start = time.time()
    nips_kmeans.converge(k=i, max_iter=20, jaccard_index_cluster_average=True, seed=None, seed_state = "random")
    end = time.time()
    print("{} clusters took {} secs time to be found".format(i,end-start))
    print("_________________________________________________________________")
    inertia_nips.append(nips_kmeans.inertia)
    
plt.plot(inertia)
plt.grid()

#If 8 is a good cluster count from the elbow curve criteria then we would like to view the most frequent words in each cluster.
nips_kmeans.converge(k=8, max_iter=50, jaccard_index_cluster_average=True, seed=None, seed_state = "random")
nips_kmeans.cluster_top_words(50)



#Training with top-k Dissimilar documents seed initialisation ------------------------------------------------

inertia_nips_d = []
for i in range(1,50):
    start = time.time()
    nips_kmeans.converge(k=i, max_iter=20, jaccard_index_cluster_average=True, seed=None, seed_state = "dissimilar")
    end = time.time()
    print("{} clusters took {} secs time to be found".format(i,end-start))
    print("_________________________________________________________________")
    inertia_nips_d.append(nips_kmeans.inertia)
    
#If 8 is a good cluster count from the elbow curve criteria then we would like to view the most frequent words in each cluster.
nips_kmeans.converge(k=8, max_iter=50, jaccard_index_cluster_average=True, seed=None, seed_state = "dissimilar")
nips_kmeans.cluster_top_words(50)
