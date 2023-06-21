import mrdja.ransac.coreransac as coreransac
import mrdja.pointcloud as pointcloud
import mrdja.geometry as geometry
import mrdja.drawing as drawing
import open3d as o3d
import numpy as np

# The files in the Stanford database are like this:
# {'number_pcd_points': 1136617, 'has_normals': False, 
# 'has_colors': True, 'is_empty': False, 'max_x': -15.207, 
# 'min_x': -20.542, 'max_y': 41.283, 'min_y': 36.802, 
# 'max_z': 3.206, 'min_z': 0.02, 'all_points_finite': True, 
# 'all_points_unique': False}
# the measures are in meters
# dict_results = pointcloud_audit(pcd)
# print(dict_results)

def get_RANSAC_data_from_file(filename, ransac_iterations, threshold, seed):
    dict_full_results = {}
    dict_full_results["filename"] = filename

    pcd = o3d.io.read_point_cloud(filename)
    '''
    audit_before_sanitizing = pointcloud.pointcloud_audit(pcd)
    dict_full_results["audit_before_sanitizing"] = audit_before_sanitizing
    pcd = pointcloud.pointcloud_sanitize(pcd)
    audit_after_sanitizing = pointcloud.pointcloud_audit(pcd)
    dict_full_results["audit_after_sanitizing"] = audit_after_sanitizing
    '''
    number_pcd_points = len(pcd.points)
    dict_full_results["number_pcd_points"] = number_pcd_points
    np_points = np.asarray(pcd.points)

    dict_full_results["ransac_iterations_results"] = []

    np.random.seed(seed)
    max_number_inliers = 0
    best_iteration_results = None
    for _ in range(ransac_iterations):
        dict_iteration_results = coreransac.get_ransac_iteration_results(np_points, threshold, number_pcd_points)
        if dict_iteration_results["number_inliers"] > max_number_inliers:
            max_number_inliers = dict_iteration_results["number_inliers"]
            best_iteration_results = dict_iteration_results
        dict_full_results["ransac_iterations_results"].append(dict_iteration_results)
    dict_full_results["ransac_best_iteration_results"] = best_iteration_results
    return dict_full_results

ransac_iterations = 1000
threshold = 0.02
seed = 42

filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/conferenceRoom_1/conferenceRoom_1.ply"
RANSAC_data_from_file = get_RANSAC_data_from_file(filename, ransac_iterations, threshold, seed)

print(RANSAC_data_from_file)

indices_inliers = RANSAC_data_from_file["ransac_best_iteration_results"]["indices_inliers"]
# create a point cloud with the inliers
pcd = o3d.io.read_point_cloud(filename)
inlier_cloud = pcd.select_by_index(indices_inliers)
o3d.visualization.draw_geometries([inlier_cloud])
# paint the inliers red
inlier_cloud.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd, inlier_cloud])

# save RANSAC data to file as pickle
import pickle
with open('RANSAC_data_from_file.pickle', 'wb') as handle:
    pickle.dump(RANSAC_data_from_file, handle, protocol=pickle.HIGHEST_PROTOCOL)

# create a np array with all the points
list_iterations_results = RANSAC_data_from_file["ransac_iterations_results"]
all_points = np.empty((0, 3))
for iteration_results in list_iterations_results:
    current_points = iteration_results["current_random_points"]
    print(current_points.shape)
    print(all_points.shape)
    all_points = np.append(all_points, current_points, axis=0)
print(all_points.shape)

# create a np array with all the planes
all_planes = np.empty((0, 4))
for iteration_results in list_iterations_results:
    current_plane = iteration_results["current_plane"]
    # convert current_plane shape from (4,) to (1, 4)
    current_plane = np.expand_dims(current_plane, axis=0)
    print(current_plane)
    print(current_plane.shape)
    print(all_planes.shape)
    all_planes = np.append(all_planes, current_plane, axis=0)
print(all_planes.shape)

# create a np array of dimensions (all_planes.shape[0], all_points.shape[0]) with the distances of each plane to all the points
all_distances = np.empty((all_planes.shape[0], all_points.shape[0]))
for i in range(all_planes.shape[0]):
    current_plane = all_planes[i]
    print(current_plane)
    print(current_plane.shape)
    current_distances = geometry.get_distance_from_points_to_plane(all_points, current_plane)
    print(current_distances.shape)
    all_distances[i] = current_distances

import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Perform hierarchical/agglomerative clustering and find the best clustering using silhouette score
best_score = -1
best_n_clusters = -1
best_labels = None
for n_clusters in range(2, 10):
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(all_distances)
    silhouette_avg = silhouette_score(all_distances, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_n_clusters = n_clusters
        best_labels = cluster_labels

# project the data points to a 2D space using PCA and color the points according to cluster labels
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(all_distances)
pca_2d = pca.transform(all_distances)
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=best_labels)
plt.show()

# project the data points to a 3D space using PCA and color the points according to cluster labels
from sklearn.decomposition import PCA
pca = PCA(n_components=3).fit(all_distances)
pca_3d = pca.transform(all_distances)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2], c=best_labels)
plt.show()

# count the number of elements in each cluster
unique, counts = np.unique(best_labels, return_counts=True)

# get the index of an element from the most populated cluster
most_populated_cluster = np.argmax(counts)
# get the first index from best_labels that has the value most_populated_cluster
indices_most_populated_cluster = np.where(best_labels == most_populated_cluster)[0]
# get the planes that correspond to the indices of the most populated cluster
planes_most_populated_cluster = all_planes[indices_most_populated_cluster]
# get the points that correspond to the indices of the most populated cluster, taking into account that 
# the points are in the same order as the planes, and there are three of them for each plane
index_first_point_most_populated_cluster = indices_most_populated_cluster * 3
# each element of points_most_populated_cluster is a np array of shape (3,) with the coordinates of the three points
# that define the plane
points_most_populated_cluster = np.empty((0, 3, 3))
for i in range(len(index_first_point_most_populated_cluster)):
    current_index_first_point = index_first_point_most_populated_cluster[i]
    current_point_first = all_points[current_index_first_point]
    current_point_second = all_points[current_index_first_point+1]
    current_point_third = all_points[current_index_first_point+2]
    current_points = np.array([[current_point_first, current_point_second, current_point_third]])
    points_most_populated_cluster = np.append(points_most_populated_cluster, current_points, axis=0)


# get the geometries of all the planes using drawing.draw_plane_as_lines_open3d(*closest_plane, line_color=[0, 1, 1])
planes_most_populated_cluster_geometries = []
for plane in planes_most_populated_cluster:
    plane_geometry = drawing.draw_plane_as_lines_open3d(*plane, line_color=[0, 1, 1])
    planes_most_populated_cluster_geometries.append(plane_geometry)
# draw the geometries of the planes along with the point cloud
o3d.visualization.draw_geometries([pcd, *planes_most_populated_cluster_geometries])
plane_geometry = drawing.draw_plane_as_lines_open3d(*planes_most_populated_cluster[0], line_color=[0, 1, 1])
o3d.visualization.draw_geometries([pcd, plane_geometry])

