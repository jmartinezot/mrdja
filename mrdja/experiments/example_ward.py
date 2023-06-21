import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Generate random data points
np.random.seed(0)
X = np.random.randn(1000, 3000)

# Perform hierarchical/agglomerative clustering and find the best clustering using silhouette score
best_score = -1
best_n_clusters = -1
best_labels = None
for n_clusters in range(2, 10):
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_n_clusters = n_clusters
        best_labels = cluster_labels

# project the data points to a 2D space using PCA and color the points according to cluster labels
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X)
pca_2d = pca.transform(X)
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=best_labels)
plt.show()


