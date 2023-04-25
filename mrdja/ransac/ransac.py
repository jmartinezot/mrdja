import open3d as o3d
import coreransac
import coreransaccuda
import random
import numpy as np
from typing import Tuple

def o3d_plane_finder(pcd: o3d.geometry.PointCloud, distance_threshold: float, 
                          num_iterations: int) -> Tuple[np.ndarray, o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    This function performs a plane segmentation using Open3D.
    It returns the plane model, the inlier point cloud, and the outlier point cloud.
    """
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=num_iterations)
    # [a, b, c, d] = plane_model
    # print("Plane equation: {}x + {}y + {}z + {} = 0".format(a, b, c, d))
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return plane_model, inlier_cloud, outlier_cloud

def ransac_from_scratch_plane_finder(pcd, distance_threshold, num_iterations):
    num_points = len(pcd.points)
    pcd_points = np.asarray(pcd.points, dtype="float32") 
    pcd_colors = np.asarray(pcd.colors, dtype="float32")
    dict_results = coreransac.get_ransac_results(pcd_points, num_points, distance_threshold, num_iterations)
    best_model = dict_results["best_plane"]
    best_inlier_indices = dict_results["indices_inliers"]
    # best_inlier_count = dict_results["number_inliers"] 
    best_inlier_cloud_points = pcd_points[best_inlier_indices]
    # fill best_outlier_cloud_points with the points that are not inliers
    best_outlier_indices = np.setdiff1d(np.arange(num_points), best_inlier_indices)
    best_outlier_cloud_points = pcd_points[best_outlier_indices]
    best_inlier_cloud_colors = pcd_colors[best_inlier_indices]
    best_outlier_cloud_colors = pcd_colors[best_outlier_indices]
    best_inlier_cloud = o3d.geometry.PointCloud()
    best_inlier_cloud.points = o3d.utility.Vector3dVector(best_inlier_cloud_points)
    best_inlier_cloud.colors = o3d.utility.Vector3dVector(best_inlier_cloud_colors)
    best_outlier_cloud = o3d.geometry.PointCloud()
    best_outlier_cloud.points = o3d.utility.Vector3dVector(best_outlier_cloud_points)
    best_outlier_cloud.colors = o3d.utility.Vector3dVector(best_outlier_cloud_colors)
            
    return best_model, best_inlier_cloud, best_outlier_cloud

def cuda_plane_finder(pcd, distance_threshold, num_iterations):
    num_points = len(pcd.points)
    pcd_points = np.asarray(pcd.points, dtype="float32") 
    pcd_colors = np.asarray(pcd.colors, dtype="float32")
    dict_results = coreransaccuda.get_ransac_results_cuda(pcd_points, num_points, distance_threshold, num_iterations)
    best_model = dict_results["best_plane"]
    best_inlier_indices = dict_results["indices_inliers"]
    # best_inlier_count = dict_results["number_inliers"] 
    best_inlier_cloud_points = pcd_points[best_inlier_indices]
    # fill best_outlier_cloud_points with the points that are not inliers
    best_outlier_indices = np.setdiff1d(np.arange(num_points), best_inlier_indices)
    best_outlier_cloud_points = pcd_points[best_outlier_indices]
    best_inlier_cloud_colors = pcd_colors[best_inlier_indices]
    best_outlier_cloud_colors = pcd_colors[best_outlier_indices]
    best_inlier_cloud = o3d.geometry.PointCloud()
    best_inlier_cloud.points = o3d.utility.Vector3dVector(best_inlier_cloud_points)
    best_inlier_cloud.colors = o3d.utility.Vector3dVector(best_inlier_cloud_colors)
    best_outlier_cloud = o3d.geometry.PointCloud()
    best_outlier_cloud.points = o3d.utility.Vector3dVector(best_outlier_cloud_points)
    best_outlier_cloud.colors = o3d.utility.Vector3dVector(best_outlier_cloud_colors)
            
    return best_model, best_inlier_cloud, best_outlier_cloud
