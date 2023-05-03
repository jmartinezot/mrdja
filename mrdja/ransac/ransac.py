import open3d as o3d
import mrdja.ransac.coreransac as coreransac
import mrdja.ransac.coreransaccuda as coreransaccuda
import random
import numpy as np
from typing import Tuple, Dict, Union

def o3d_plane_finder(pcd: o3d.geometry.PointCloud, distance_threshold: float, 
                          num_iterations: int) -> Dict[str, Union[np.ndarray, o3d.geometry.PointCloud]]:
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
    return {"plane_model": plane_model, "inlier_cloud": inlier_cloud, "outlier_cloud": outlier_cloud}

def ransac_from_scratch_plane_finder(pcd: o3d.geometry.PointCloud, 
                                      distance_threshold: float, 
                                      num_iterations: int) -> Dict[str, Union[np.ndarray, o3d.geometry.PointCloud]]:
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
    dict_results = coreransac.get_ransac_results(pcd_points, num_points, distance_threshold, num_iterations)
    plane_model = dict_results["best_plane"]
    best_inlier_indices = dict_results["indices_inliers"]   
    inlier_cloud = pcd.select_by_index(best_inlier_indices)
    outlier_cloud = pcd.select_by_index(best_inlier_indices, invert=True)
            
    return {"plane_model": plane_model, "inlier_cloud": inlier_cloud, "outlier_cloud": outlier_cloud}

def cuda_plane_finder(pcd: o3d.geometry.PointCloud, 
                      distance_threshold: np.float32, 
                      num_iterations: np.int64) -> Dict[str, Union[np.ndarray, o3d.geometry.PointCloud]]:
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
    if num_points == 0:
        raise ValueError("Input point cloud is empty")   
    pcd_points = np.asarray(pcd.points, dtype="float32") 
    dict_results = coreransaccuda.get_ransac_results_cuda(pcd_points, num_points, distance_threshold, num_iterations)
    plane_model = dict_results["best_plane"]
    best_inlier_indices = dict_results["indices_inliers"]   
    inlier_cloud = pcd.select_by_index(best_inlier_indices)
    outlier_cloud = pcd.select_by_index(best_inlier_indices, invert=True)
            
    return {"plane_model": plane_model, "inlier_cloud": inlier_cloud, "outlier_cloud": outlier_cloud}

def find_multiple_planes(pcd: o3d.geometry.PointCloud, distance_threshold: float, num_iterations: int, num_planes: int, finder_function) -> Dict[str, Union[np.ndarray, o3d.geometry.PointCloud]]:
    planes = []
    for _ in range(num_planes):
        if len(pcd.points) < 3:
            break
        d = finder_function(pcd, distance_threshold, num_iterations)
        planes.append(d)
        pcd = d["outlier_cloud"]
    return planes

def color_multiple_planes(pcd: o3d.geometry.PointCloud, distance_threshold: float, num_iterations: int, num_planes: int, finder_function) -> o3d.geometry.PointCloud:
    planes = find_multiple_planes(pcd, distance_threshold, num_iterations, num_planes, finder_function)
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    colored_pcd = o3d.geometry.PointCloud()
    for i, plane in enumerate(planes):
        color = colors[i % len(colors)]
        plane["inlier_cloud"].paint_uniform_color(color)
        colored_pcd += plane["inlier_cloud"]
    colored_pcd += planes[-1]["outlier_cloud"]
    return colored_pcd
