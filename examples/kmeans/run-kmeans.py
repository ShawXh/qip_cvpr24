import numpy as np
import argparse
import torch, torch_qip

parser = argparse.ArgumentParser(description="MNIST Task")
parser.add_argument("--num-qubits", type=int, default=6)
parser.add_argument("--sample-times", type=int, default=1)
parser.add_argument("--out-strategy", type=str, default="mode", choices=["mode", "avg"])
parser.add_argument("--mode", type=str, default="classical", choices=["classical", "quantum"])
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
print(args)

np.random.seed(args.seed)

# randomly initializing K centroid by picking K samples from X
def initialize_random_centroids(K, X):
    """Initializes and returns k random centroids"""
    m, n = np.shape(X)
    # a centroid should be of shape (1, n), so the centroids array will be of shape (K, n)
    centroids = np.empty((K, n))
    for i in range(K):
        # pick a random data point from X as the centroid
        centroids[i] =  X[np.random.choice(range(m))] 
    return centroids

def euclidean_distance(x1, x2):
    """Calculates and returns the euclidean distance between two vectors x1 and x2"""
    return np.sqrt(np.sum(np.power(x1 - x2, 2)))

def quantum_euclidean_distance(x1, x2):
    """Calculates and returns the euclidean distance between two vectors x1 and x2"""
    x1_length = np.sqrt(np.sum(np.power(x1, 2)))
    x2_legnth = np.sqrt(np.sum(np.power(x2, 2)))
    inner_prod = torch_qip.elementqip(
        args.num_qubits,
        np.dot(x1 / x1_length, x2 / x2_legnth),
        args.sample_times,
        False,
        args.out_strategy,
    )
    return np.sqrt(x1_length * x1_length + x2_legnth * x2_legnth\
                    - 2 * inner_prod * x1_length * x2_legnth)

def closest_centroid(x, centroids, K):
    """Finds and returns the index of the closest centroid for a given vector x"""
    distances = np.empty(K)
    for i in range(K):
        if (args.mode == "classical"):
            distances[i] = euclidean_distance(centroids[i], x)
        else:
            distances[i] = quantum_euclidean_distance(centroids[i], x)
    return np.argmin(distances) # return the index of the lowest distance

def create_clusters(centroids, K, X):
    """Returns an array of cluster indices for all the data samples"""
    m, _ = np.shape(X)
    cluster_idx = np.empty(m)
    for i in range(m):
        cluster_idx[i] = closest_centroid(X[i], centroids, K)
    return cluster_idx

def compute_means(cluster_idx, K, X):
    """Computes and returns the new centroids of the clusters"""
    _, n = np.shape(X)
    centroids = np.empty((K, n))
    for i in range(K):
        points = X[cluster_idx == i] # gather points for the cluster i
        centroids[i] = np.mean(points, axis=0) # use axis=0 to compute means across points
    return centroids

def run_Kmeans(K, X, max_iterations=500):
    """Runs the K-means algorithm and computes the final clusters"""
    # initialize random centroids
    centroids = initialize_random_centroids(K, X)
    # loop till max_iterations or convergance
    # print(f"initial centroids: {centroids}")
    for _ in range(max_iterations):
        print(_)
        # create clusters by assigning the samples to the closet centroids
        clusters = create_clusters(centroids, K, X)
        previous_centroids = centroids                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        # compute means of the clusters and assign to centroids
        centroids = compute_means(clusters, K, X)
        # if the new_centroids are the same as the old centroids, return clusters
        diff = previous_centroids - centroids
        if not diff.any():
            return clusters
    return clusters

from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from scipy.misc import comb
from itertools import combinations
import numpy as np

def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)    
    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred

def rand_score (labels_true, labels_pred):
    check_clusterings(labels_true, labels_pred)
    my_pair = list(combinations(range(len(labels_true)), 2)) #create list of all combinations with the length of labels.
    def is_equal(x):
        return (x[0]==x[1])
    my_a = 0
    my_b = 0
    for i in range(len(my_pair)):
            if(is_equal((labels_true[my_pair[i][0]],labels_true[my_pair[i][1]])) == is_equal((labels_pred[my_pair[i][0]],labels_pred[my_pair[i][1]])) 
               and is_equal((labels_pred[my_pair[i][0]],labels_pred[my_pair[i][1]])) == True):
                my_a += 1
            if(is_equal((labels_true[my_pair[i][0]],labels_true[my_pair[i][1]])) == is_equal((labels_pred[my_pair[i][0]],labels_pred[my_pair[i][1]])) 
               and is_equal((labels_pred[my_pair[i][0]],labels_pred[my_pair[i][1]])) == False):
                my_b += 1
    my_denom = comb(len(labels_true),2)
    ri = (my_a + my_b) / my_denom
    return ri

data, labels = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size
y_preds = run_Kmeans(n_digits, data)
nmi = normalized_mutual_info_score(labels, y_preds)
ami = adjusted_mutual_info_score(labels, y_preds)
ri = rand_score(labels, y_preds)
# print("NMI:", normalized_mutual_info_score(labels, y_preds))
# print("AMI:", adjusted_mutual_info_score(labels, y_preds))
# print("RI:", rand_score(labels, y_preds))

with open("log-cluster.txt", "a") as f:
    f.write(str(args) + "\n")
    f.write("NMI: {nmi}, AMI: {ami}, RI: {ri}".format(
        nmi=nmi, ami=ami, ri=ri))
    f.write("\n")

# print("acc: ", (labels == y_preds).sum() / len(labels))

# print(normalized_mutual_info_score(labels, y_preds))
# creating a dataset for clustering
# X, y = datasets.make_blobs(
#     n_features=100,
#     n_samples=10000,
#     random_state=0,
# )
# y_preds = run_Kmeans(3, X)
# print("NMI:", normalized_mutual_info_score(y, y_preds))


# kmeans = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)