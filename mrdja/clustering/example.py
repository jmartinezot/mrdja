from mrdja.clustering import clustering_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate n-dimensional sample data using make_blobs
n_samples = 100  # Number of data points
n_features = 300    # Number of dimensions
centers = 3       # Number of clusters
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=42)

# perform PCA to reduce the dimensionality of the data
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# X = pca.fit_transform(X)

# Perform clustering with K-means, using some criteria to choose the right number of clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
scores = []
max_clusters = 15
for k in range(2, max_clusters):
    clustering = KMeans(n_clusters=k)
    clustering.fit(X)
    score = silhouette_score(X, clustering.labels_)
    scores.append(score)
plt.plot(range(2, max_clusters), scores)
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.show()
# choose the number of clusters that maximizes the silhouette score
# compute the number of clusters that maximizes the silhouette score
n_clusters = np.argmax(scores) + 2
clustering = KMeans(n_clusters=n_clusters)
clustering.fit(X)

# perform PCA to reduce the dimensionality of the data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pca.fit_transform(X)
# plot the data, writing the percent of variance explained by each principal component
plt.scatter(X[:,0], X[:,1], c=clustering.labels_)
plt.xlabel("PC1 ({}%)".format(round(pca.explained_variance_ratio_[0]*100, 2)))
plt.ylabel("PC2 ({}%)".format(round(pca.explained_variance_ratio_[1]*100, 2)))
plt.show()

result, report = clustering_report(X)
