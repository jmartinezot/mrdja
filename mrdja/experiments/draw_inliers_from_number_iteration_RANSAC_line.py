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
        dict_iteration_results = coreransac.get_ransac_line_iteration_results(np_points, threshold, number_pcd_points)
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

# create a function that takes the RANSAC_data_from_file and the iteration number and returns the inliers
# create a function that takes the RANSAC_data_from_file and the iteration number and returns the number of inliers

pcd = o3d.io.read_point_cloud(RANSAC_data_from_file["filename"])
np_points = np.asarray(pcd.points)

def get_inliers_from_iteration(RANSAC_data_from_file, np_points, iteration_number):
    dict_iteration_results = RANSAC_data_from_file["ransac_iterations_results"][iteration_number]
    inliers = np_points[dict_iteration_results["indices_inliers"]]
    pcd_inliers = o3d.geometry.PointCloud()
    pcd_inliers.points = o3d.utility.Vector3dVector(inliers)
    pcd_inliers.paint_uniform_color([1, 0, 0])
    return pcd_inliers

def get_number_inliers_from_iteration(RANSAC_data_from_file, iteration_number):
    dict_iteration_results = RANSAC_data_from_file["ransac_iterations_results"][iteration_number]
    return dict_iteration_results["number_inliers"]

# using open3d create a geometry with the inliers painted red, and show the inliers, the original point cloud and the number of iteration and the number
# of inliers in the same window; create a for loop that shows the inliers for each iteration

def draw_inliers_from_iteration(RANSAC_data_from_file, pcd, iteration_number):
    pcd_inliers = get_inliers_from_iteration(RANSAC_data_from_file, np_points, iteration_number)
    number_inliers = get_number_inliers_from_iteration(RANSAC_data_from_file, iteration_number)
    o3d.visualization.draw_geometries([pcd, pcd_inliers], window_name="Inliers from iteration " + str(iteration_number) + " with " + str(number_inliers) + " inliers")

for iteration_number in range(ransac_iterations):
    # only show if the number of inliers is greater than threshold
    threshold = 1000
    if get_number_inliers_from_iteration(RANSAC_data_from_file, iteration_number) > threshold:
        draw_inliers_from_iteration(RANSAC_data_from_file, pcd, iteration_number)





