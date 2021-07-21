# Jaccard_Similarity_based_K-Means 
[Find the data here](https://archive.ics.uci.edu/ml/datasets/Bag+of+Words): The 5 datasets are of different sizes. I have tried my methodology on the 3 smaller datasets which are Enron emails, NIPS blog entries, KOS blog entries. The other 2 datasets are NYTimes news articles and PubMed abstracts which are huge and they would need powerful resources to perform clustering.


----------------------------------------------------------

# There are two approaches that have been attempted to apply K-means clustering using jaccard index:

## K-Clusters: 
> There is no single mean value in any cluster. Rather a documents affinity to a particular cluster is judged by the average of document's jaccard similarity with all other documents in the cluster. (Tried on: KOS, NIPS)

> Here, inertia is the sum of the average similarity of all the documents with all its other cluster documents.

> K-Clusters which takes *quadratic time* for each iteraion is not feasible for classification of Enron collection as the number of documents is high in it.

It is observed that update of every 100 documents jaccard indices with all other documents (40000 times 100 operations) takes 30 seconds on average. This means that 40000 updates of jaccard matrix would consume approximately (4 times 10000 times 30)/100 seconds (= 25hrs approx.) which is a large amount of time span and extremely consuming using local resources.

## K-means: 
> There is a mean associated with each cluster. The mean of a cluster is the document in the cluster whose jaccard similarity is highest with the arithmetic mean of all documents in the cluster. Here, the arithmetic mean refers to the average of the frequency for each word in the vocabulary for the cluster. 

> The mean for each word is between 0 to 1. So, if the frequency for the word is more than 0.5, then we consider the word to be in the mean document else we discard it (Tried on: ENRON). Then, associating the documents to the best cluster is done by seeing the documents jaccard similarity with the mean document.

> The inertia is sum of the jaccard similarity of the documents with their cluster representative documents (i.e, the most similar document to the arithmetic mean of the cluster)

> Here, the *Time Complexity* of each iteration in finding the cluster means and the documents is *BigOh(n)*.
----------------------------------------------------------

