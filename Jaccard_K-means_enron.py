#
__author__ = "Chandrashish Prasad"
__license__ = "Feel free to copy, I appreciate if you learn from here"

#READING

import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import time

#Reading the document ---------------------------------------------

fh = open("docword.enron.txt", 'r')
doc_count = int(fh.readline().strip())
vocab_count = int(fh.readline().strip())
entries = int(fh.readline().strip())
lines = fh.readlines()
doc_list_voc = {}    #to store the docs as list of words with ith index bein 1 if word in the doc
for i in range(1,doc_count+1):
    doc_list_voc[i] = [0]*(vocab_count+1)
doc={}               #to store the docs as set of words
i = 0
print("Reading the Documents:")
for line in lines:
    a = list(map(int, line.strip().split()))
    i+=1
    if len(a)>1:
        doc_id, word_id, freq = a
        if doc_id not in doc.keys():
            doc[doc_id] = set([word_id])
        else:
            doc[doc_id].add(word_id)
        doc_list_voc[doc_id][word_id]=1
    if(i%500000==0):
        print("{}th line read".format(i))
fh.close()

#Needed Functions -------------------------------------------------
def jaccard(set1, set2):
    index = float(len(set1.intersection(set2)))/float(len(set1.union(set2)))
    return index, 1-index

def get_seeds(doc_count, k):
    return list(np.random.choice(range(1,doc_count+1), k, replace=False))
       
def clusterize(doc_count, k, seed, doc_list_voc, doc):       #to initialise the cluster_to_id and id_cluster dictionaries using the seeds provided
    clusters = {}
    rev_clusters = {}
    for i in range(1,doc_count+1):
        rev_clusters[i] = -1
    for i in range(k):
        clusters[i] = set([seed[i]])
        rev_clusters[seed[i]] = i 
        #Now associate each doc to a cluster by seeing jac_similarity with the seed of the cluster
    inv_inertia = 0
    for i in range(1, doc_count+1):
        if i in seed:
            continue
        temp_clust = -1
        temp_ind = -1
        for j in range(k):
            ind, dist = jaccard(doc[i],doc[seed[j]])
            if ind>temp_ind:
                temp_ind = ind
                temp_clust = j
        inv_inertia += temp_ind
        clusters[temp_clust].add(i)
        rev_clusters[i] = temp_clust
    return clusters, rev_clusters, inv_inertia

def find_doc(cluster, mean_set, doc):
    temp_sim = -1
    temp_doc = -1
    for i in cluster:
        sim_ind, sim_dis = jaccard(mean_set, doc[i])
        if sim_ind>temp_sim:
            temp_sim = sim_ind
            temp_doc = i
    return i
        
def find_new_seeds(doc_count, k, seed, doc_list_voc, doc, clusters, rev_clusters, vocab_count):
    new_seed = []
    for j in range(k):     #k times
        mean = [0]*(vocab_count+1)
        counter = 0
        for m in clusters[j]:      
            mean = [mean[k] + doc_list_voc[m][k] for k in range(vocab_count+1)]  #equivalent to doc_count*vocab_count considering all clusters
            counter += 1
        mean = [1 if (i/counter)>0.5 else 0 for i in mean]
        mean_set = set()
        for i in range(1,len(mean)):
            if mean[i]==1:
                mean_set.add(i)
        mean_doc = find_doc(clusters[j], mean_set, doc)
        new_seed.append(mean_doc)
    return new_seed

def jaccard_clustering(k, doc_count, iters, doc_list_voc, doc, vocab_count):
    seed = get_seeds(doc_count, k)
    inv_inertia = 0
    for i in range(iters):
        start = time.time()
        print("Seed is: ",set(seed))
        clus,rev_clus,inv_inertia = clusterize(doc_count, k, seed, doc_list_voc, doc)
        new_seed = find_new_seeds(doc_count, k, seed, doc_list_voc, doc, clus, rev_clus, vocab_count)
        end = time.time()
        if(set(seed) == set(new_seed)):
            print("{} iterations done in {} secs".format(i+1,end-start))
            print("Complete Convergence Achieved and inertia is: ",inv_inertia)
            break
        seed = new_seed
        print("{} iterations done in {} secs and the inertia is {}".format(i+1,end-start,inv_inertia))
        print("--------------------------------------------------------------------------------")
    return seed, inv_inertia

#Running the model --------------------------------------------
inv = [0,]
for k in range(1,22,3):
    start = time.time()
    final_seed, inv_inertia = jaccard_clustering(k, doc_count, 10 , doc_list_voc, doc, vocab_count)
    end = time.time()
    inv.append(inv_inertia)
    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    print("Time taken for clustering with {} clusters is {} min.".format(k,(end-start)/60))
    print("_______________________________________________________________________________________________")
    
plt.plot(inv)
plt.grid()

#It is recomended for you to try the different cluster counts and max_iterations for better results. Since, the Enron collection has almost 40,000 documents, it will take almost 50 minutes on an average for each cluster count to converge with maximum iterations set to 20.
