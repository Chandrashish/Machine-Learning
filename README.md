# Jaccard_Similarity_based_K-Means 
[Find the data here](https://archive.ics.uci.edu/ml/datasets/Bag+of+Words): The datasets are of different sizes. I have tried my methodology on the three smaller datasets (Enron emails, NIPS blog entries, KOS blog entries).


----------------------------------------------------------

# There are two approaches that have been attempted to apply K-means clustering using jaccard index:
## K-Clusters: 
### There is no single mean value in any cluster. Rather a documents affinity to a particular cluster is judged by the average of document's jaccard similarity with all other documents in the cluster. (Tried on: KOS,NIPS)
### Here, inertia is the sum of the average similarity of all the documents with all its other cluster documents.
## K-means: There is a mean associated with each cluster. The mean of a cluster is the document in the cluster whose jaccard similarity is highest with the arithmetic mean of all documents in the cluster. Here, the arithmetic mean refers to the average of the frequency for each word in the vocabulary for the cluster. The mean for each word is between 0 to 1. So, if the frequency for the word is more than 0.5, then we consider the word to be in the mean document else we discard it (Tried on: ENRON). Then, associating the documents to the best cluster is done by seeing the documents jaccard similarity with the mean document.

----------------------------------------------------------

