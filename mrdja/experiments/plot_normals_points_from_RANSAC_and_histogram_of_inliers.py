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
threshold = 0.002
seed = 42

filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/conferenceRoom_1/conferenceRoom_1.ply"
RANSAC_data_from_file = get_RANSAC_data_from_file(filename, ransac_iterations, threshold, seed)

print(RANSAC_data_from_file)

# create a np array with all the planes
list_iterations_results = RANSAC_data_from_file["ransac_iterations_results"]
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

# create a list with all the numbers of inliers
list_number_inliers = []
for iteration_results in list_iterations_results:
    list_number_inliers.append(iteration_results["number_inliers"])
print(list_number_inliers)

# plot a histogram of the number of inliers
import matplotlib.pyplot as plt
plt.hist(list_number_inliers, bins=100)
plt.show()

import mrdja.ransac.coreransacutils as crsu
inliers_ratio = list_number_inliers[0] / RANSAC_data_from_file["number_pcd_points"]
number_iterations = crsu.compute_number_iterations(inliers_ratio, alpha=0.95)
print(number_iterations)

# get the maximum number of inliers and the position of the iteration that produced it
max_number_inliers = 0
max_number_inliers_index = 0
for index, number_inliers in enumerate(list_number_inliers):
    if number_inliers > max_number_inliers:
        max_number_inliers = number_inliers
        max_number_inliers_index = index

# compute the number of iterations needed to get a probability of 0.95 of finding the best model
inliers_ratio = max_number_inliers / RANSAC_data_from_file["number_pcd_points"]
number_iterations = crsu.compute_number_iterations(inliers_ratio, alpha=0.99)
print(number_iterations)

# extract normalized normals from the planes, and also the corresponding D parameter after normalization
all_normals = all_planes[:, 0:3]
print(all_normals.shape)
all_D = all_planes[:, 3]
all_D = all_D / np.linalg.norm(all_normals, axis=1)
all_normals = all_normals / np.linalg.norm(all_normals, axis=1)[:, None]
print(all_normals.shape)

# plot the normals as points in 3D space using matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = all_normals[:, 0]
y = all_normals[:, 1]
z = all_normals[:, 2]

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# plot a histogram of all_D values
plt.hist(all_D, bins=100)
plt.show()







