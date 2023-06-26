'''
RANSAC functions

'''

import open3d as o3d
import mrdja.ransac.coreransac as coreransac
import mrdja.ransac.coreransaccuda as coreransaccuda
import mrdja.geometry as geometry
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

def get_plane_distances_to_points(plane: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Computes the distances from a plane to a collection of points.

    :param plane: Plane parameters.
    :type plane: np.ndarray
    :param points: Collection of points.
    :type points: np.ndarray
    :return: Distances from the plane to the points.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import mrdja.ransac as ran
        >>> import numpy as np
        >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="float32")
        >>> plane = np.array([0, 0, 1, 0], dtype="float32")
        >>> ran.get_plane_distances_to_points(plane, points)
        array([ 0.,  0.,  0.], dtype=float32)
    """
    '''
    if it is a list convert it to numpy array, and check if it is empty
    '''
    if isinstance(points, list):
        points = np.asarray(points)
    if len(points) == 0:
        return np.array([], dtype="float32")
    return np.abs(np.dot(points, plane[:3]) + plane[3]) / np.linalg.norm(plane[:3])

def generate_k_planes_from_k_sets_of_3_random_points_of_pointcloud_points_and_compute_the_distance_from_the_planes_to_all_the_points(pcd: o3d.geometry.PointCloud, k: int) -> np.ndarray:
    """
    Generates k planes from k sets of 3 random points of point cloud points and computes the distance from the planes to all the points.

    :param pcd: Point cloud data.
    :type pcd: o3d.geometry.PointCloud
    :param k: Number of planes to generate.
    :type k: int
    :return: Distances from the planes to the points.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import mrdja.ransac as ran
        >>> import numpy as np
        >>> import open3d as o3d
        >>> pcd = o3d.geometry.PointCloud()
        >>> pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 2, 3], [5, 6, 7]], dtype="float32"))
        >>> ran.generate_k_planes_from_k_sets_of_3_random_points_of_pointcloud_points_and_compute_the_distance_from_the_planes_to_all_the_points(pcd, 3)
    """
    num_points = len(pcd.points)
    pcd_points = np.asarray(pcd.points, dtype="float32")
    points_to_compute_distance = np.empty((0, 3), dtype="float32")
    all_planes = []
    dict_distances = {}
    for _ in range(k):
        current_points = coreransac.get_np_array_of_three_random_points_from_np_array_of_points(pcd_points, num_points)
        current_plane = geometry.get_plane_from_list_of_three_points(current_points)
        get_plane_distances_to_points(current_plane, points_to_compute_distance)
        dict_distances[tuple(current_plane)] = get_plane_distances_to_points(current_plane, points_to_compute_distance)
        all_planes.append(current_plane)
        for plane in all_planes:
            previous_distances = dict_distances[tuple(plane)]
            current_distances = get_plane_distances_to_points(plane, current_points)
            all_distances = np.concatenate((previous_distances, current_distances))
            # convert plane to tuple to use it as key in dictionary
            dict_distances[tuple(plane)] = all_distances
        points_to_compute_distance = np.concatenate((points_to_compute_distance, current_points))
    return dict_distances
