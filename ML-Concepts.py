
# ============================================= K-Nearest Neighbor (KNN) Algorithm ====================================================
# Purpose: To make classifications or predictions about the grouping of individual data point using proximity.
# Intuition: Similar points can be found near each other, "birds of a feather flock together"
# Type: Non-parametric Supervised learning (does not make any underlying assumptions about the distribution of the data)
# Application Areas: Relevance ranking, similarity search for image and videos, pattern recognition, finance (stock market prediction)
# Implementation Details:
#       => k represents the number of nearest neighbors considered in the classification or regression problem.
#           => SMALL k values: make predictions unstable, high variance, overfitting, high complexity, low bias, fitting the training data
#           => BIG k values: noisy, increase the accuracy of the predictions, low variance, low complexity, high bias, not complex enough to fit
#           => Recommended to pick an odd value for k to avoid ties in classification analysis.

#       => 4 types of different metrics that can be used: Euclidean distance, Manhattan distance, Minkowski distance, Hamming distance
#           => Euclidean distance: shortest distance between any two points, d = √[ (x_2 – x_1)^2 + (y_2 – y_1)^2]
#           => Manhattan distance: absolute value between two points, |x_1 - x_2| + |y_1 - y_2|
#           => Minkowski distance: metric in a normed vector space, when p=1 it is manhattan distance, p=2 is euclidean, sum( [x_i - y_i] ^ p)^(1/p)
#           => Hamming distance: overlap metric, used in Boolean or string vectors to identify the differences
#       => (1) Select optimal value for K || (2) Calculate distance || (3) Find Nearest Neighbor || (4) Classification/Regression 
# Limitations: Difficulty to scale (takes too much memory and data storage), curse of dimensionality (training examples is challenged by an
#              increasing number of dimensions and the inherent increase of feature values in these dimensions), overfitting (small k), underfitting (larger k)
# =======================================================================================================================================


# ================================================ K-Means==================================================================================
# Purpose: Grouping unlabeled data points into groups or clusters. Minimize the sum of distances between datapoints and their assigned clusters.
# Intuition: Exlusive or "hard" clustering methods, a data can only exist in one of the groups.
# Type: Unsupervised clustering method
# Application Areas: Market segmentation, document clustering, image segmentation, image compression, delivery route optimization, targeting clients
# Implementation Details:
#       => Iterative method, centoid-based algorithm that partitions a dataset into similar groups based on the distance between their centroids.
#       => Centorids are "cluster centers", selected from either the mean or the median of all the points wihtin the cluster
#       => k is the number of clusters/centroids. Large K: signifies smaller clusters with greater detail && Small K: larger clusters with less detail
#       => Usually Euclidean distance is chosen as the distance metric.
#       => Maximization step computes the mean of all points for each cluster adn re-assigns the centorids accordingly. 
#       => Repeat the steps until reached convergence or maximum number of iteration is achieved.
#       => The more compact and distinct each cluster is, the better.
#       => Evaluation metric: calculating sum of squared errors (SSE) of each point to its closest centroid evaluates the quality of the cluster
#               assignments by measuring the total variation within each cluster. 
# Stepwise:
#   (1) Choose the number of clusters, k, that you want to create.
#   (2) Initialize k cluster centroids randomly.
#   (3) Assign each data point to the nearest centroid, creating k clusters.
#   (4) Recalculate the centroids as the mean of all data points in each cluster.
#   (5) Repeat steps 3 and 4 until convergence (centroids no longer change significantly) or for a specified number of iterations.
# Inertia >> minimization of the intracluster distance, lower sum is better
# Dunn Index >> minimum intercluster distance and maximum intraluster distance. High interclusters proves how different each cluster is.
# Limitations: High dependence on the input parameters, significant outlier impact, does poorly when data contains many variations and highly dimensional.
# ===========================================================================================================================================
