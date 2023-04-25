import open3d as o3d
import coreransac
import coreransaccuda
import random
import numpy as np
from typing import Tuple

def o3d_plane_finder(pcd: o3d.geometry.PointCloud, distance_threshold: float, 
                          num_iterations: int) -> Tuple[np.ndarray, o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Uses Open3D to find the best plane that fits a point cloud using RANSAC algorithm.

    :param pcd: Point cloud to fit a plane.
    :type pcd: o3d.geometry.PointCloud
    :param distance_threshold: Maximum distance a point can have to be considered in the plane.
    :type distance_threshold: float
    :param num_iterations: Number of iterations to find the plane.
    :type num_iterations: int
    :return: Parameters of the best plane model and two point clouds containing the inliers and outliers, respectively.
    :rtype: Tuple[np.ndarray, o3d.geometry.PointCloud, o3d.geometry.PointCloud]
    """
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=num_iterations)
    # [a, b, c, d] = plane_model
    # print("Plane equation: {}x + {}y + {}z + {} = 0".format(a, b, c, d))
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return plane_model, inlier_cloud, outlier_cloud

def ransac_from_scratch_plane_finder(pcd: o3d.geometry.PointCloud, 
                                      distance_threshold: float, 
                                      num_iterations: int) -> Tuple[np.ndarray, 
                                                                   o3d.geometry.PointCloud, 
                                                                   o3d.geometry.PointCloud]:
    """
    Computes the best plane that fits a collection of points and the indices of the inliers.
    
    :param pcd: The point cloud data.
    :type pcd: o3d.geometry.PointCloud
    :param distance_threshold: Maximum distance to the plane.
    :type distance_threshold: float
    :param num_iterations: Number of iterations to compute the best plane.
    :type num_iterations: int
    :return: Best plane parameters and the indices of the points that fit the plane.
    :rtype: Tuple[np.ndarray, o3d.geometry.PointCloud, o3d.geometry.PointCloud]
    """
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

def cuda_plane_finder(pcd: o3d.geometry.PointCloud, 
                      distance_threshold: np.float32, 
                      num_iterations: np.int64) -> Tuple[np.ndarray, o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Computes the best plane that fits a point cloud and the indices of the inliers using GPU acceleration.
    
    :param pcd: The point cloud.
    :type pcd: o3d.geometry.PointCloud
    :param distance_threshold: Maximum distance to the plane.
    :type distance_threshold: np.float32
    :param num_iterations: Number of iterations to compute the best plane.
    :type num_iterations: np.int64
    :return: The best plane parameters, the inliers point cloud and the outliers point cloud.
    :rtype: Tuple[np.ndarray, o3d.geometry.PointCloud, o3d.geometry.PointCloud]
    """
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
