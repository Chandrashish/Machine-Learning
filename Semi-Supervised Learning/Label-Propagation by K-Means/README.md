# Label-Propagation by K-Means:
Here, I have attempted to showcase one of the very commonly used technique in supervised learning where we label the unlabelled data by clustering and then train on the newly generated data. The data used here is Fashion-MNIST from keras which has 10 classes. It can be seen [here](https://keras.io/api/datasets/fashion_mnist/)

#### Note: Although Fashion-MNIST is a labelled data set, I have attempted to overwrite the labels while propogating the cluster centre labels and then trained a neural network for this new data-set. So, in case we had unlabelled data, we could have still used this technique.

## The entire project is structured as:
> Firstly, we attempt to find an optimum number of clusters using k-means clustering and [elbow curve](https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/)/[silhouette-score](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
>> * This exercise is done just to indicate that even though we know that there are 10 classes in the data, the above method suggests 7 as a possible optimum cluster count as well as some larger number of clusters.
>> * An important observation is at cluster count 5, where we again perform clustering on some of the clusters already found at cluster count 5.
>> * While doing this task, I also plotted some histograms on categories in test data to visualise the performance of clustering at a particular cluster count (5 & 7 here).

> Next, in line we have trained a neural network on the entire training data and tested its performance on the test-set to get a test accuracy of 89.6% in 30 epochs. 
>>  * Now, here we can expect that this result will be better than the upcoming experiments since we have trained the network with entire training data as well as with the correct label for each data point. But in a real practical scenario, we will be dealing with unlabelled data. So, the advantage of seeing the below exercise has to be understood keeping this in mind. Let's see what happens next!

> Now to exhibit the advantage of clustering, we first retrain the network on 100 randomly selected images and their labels to get a test accuracy of 61.3% in 30 epochs. However, we get slightly better results if we train the network on 100 cluster centres after performing k-means clustering on the training data with the test accuracy jumping to 63.4%.
>> * Here, the 100 cluster centres can be considered as representative images of their clusters and are distinctive from each other leading to improvement in accuracy as seen above.

> Moving a step ahead if we propagate the 100 cluster centre's labels to 70% of data in each cluster and then train the network on this newly crafted dataset, we observe that the test accuracy jumps to 71.52%. 
>> * But, propagating the labels to 100% data could lower the test accuracy as some overlaps or outliers while clustering could have been assigned wrong labels.

* Overall, we usually come across many situations where we have unlabelled data and due to its huge volume it gets very expensive to label them manually. Unsupervised learning techniques are very important in such cases to rescue us!!!
