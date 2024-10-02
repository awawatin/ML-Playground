
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
# Purpose:
# Intuition:
# Type:
# Application Areas:
# Implementation Details:
# Limitations:
# ===========================================================================================================================================
