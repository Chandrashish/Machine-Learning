# Label-Propagation by K-Means:
Here, I have attempted to showcase one of the very commonly used technique in supervised learning where we label the unlabelled data by clustering and then train on the newly generated data. The data used here is Fashion-MNIST from keras which has 10 classes. It can be seen [here](https://keras.io/api/datasets/fashion_mnist/)

### Note: Although Fashion-MNIST is a labelled data set, I have attempted to overwrite the labels while propogating the cluster centre labels and then trained a neural network for this new data-set. So, in case we had unlabelled data, we could have still used this technique.

The entire project is structured as:
*
