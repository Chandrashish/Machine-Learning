# Label-Propagation by K-Means:
Here, I have attempted to showcase one of the very commonly used technique in supervised learning where we label the unlabelled data by clustering and then train on the newly generated data. The data used here is Fashion-MNIST from keras which has 10 classes. It can be seen [here](https://keras.io/api/datasets/fashion_mnist/)

#### Note: Although Fashion-MNIST is a labelled data set, I have attempted to overwrite the labels while propogating the cluster centre labels and then trained a neural network for this new data-set. So, in case we had unlabelled data, we could have still used this technique.

## The entire project is structured as:
> Firstly, we attempt to find an optimum number of clusters using k-means clustering and [elbow curve](https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/)/[silhouette-score](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
