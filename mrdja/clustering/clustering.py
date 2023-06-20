import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt

def run_clustering(algorithm, data, **kwargs):
    # Initialize the clustering algorithm
    clustering = algorithm(**kwargs)
    
    # Perform clustering on the data
    labels = clustering.fit_predict(data)
    
    # Return the cluster labels
    return labels

def clustering_report(data):
    # Perform clustering with different algorithms and parameters
    kmeans_labels = run_clustering(KMeans, data, n_clusters=3)
    dbscan_labels = run_clustering(DBSCAN, data, eps=0.5, min_samples=5)
    agg_labels = run_clustering(AgglomerativeClustering, data, n_clusters=3)
    
    # Create a DataFrame to store the results
    results_df = pd.DataFrame({"KMeans": kmeans_labels, "DBSCAN": dbscan_labels, "Agglomerative": agg_labels})
    
    # Generate descriptive statistics for each cluster algorithm
    report = results_df.describe()
    
    # Print the report
    return results_df, report


