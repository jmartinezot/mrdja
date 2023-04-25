'''

FIX ALL DOCSTRINGS

import open3d as o3d
import numpy as np
import coreransac as crs
import coreransaccuda as crscuda

def remove_origin_points(pcd):
    """
    Removes all points with (0, 0, 0) coordinates from an Open3D point cloud.
    """
    new_pcd = o3d.geometry.PointCloud()
    points = np.asarray(pcd.points)
    new_points = points[~np.all(points == 0, axis=1)]
    colors = np.asarray(pcd.colors)
    new_colors = colors[~np.all(points == 0, axis=1)]
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    new_pcd.colors = o3d.utility.Vector3dVector(new_colors)
    return new_pcd


pcd_filename = "/home/scpmaotj/Lanverso/12-16-2022/free/p1.ply"
pcd = o3d.io.read_point_cloud(pcd_filename)
o3d.visualization.draw_geometries([pcd])
pcd = remove_origin_points(pcd)
o3d.visualization.draw_geometries([pcd])
pcd_points = np.asarray(pcd.points, dtype="float32") 
num_points = len(pcd_points)
threshold = 2
num_iterations = 30
dict_results = crs.get_ransac_results(pcd_points, num_points, threshold, num_iterations)
# dict_results = crscuda.get_ransac_results_cuda(pcd_points, num_points, threshold, num_iterations)
inliers = dict_results["indices_inliers"]
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
'''

import numpy as np
import open3d as o3d
import math
import coreransacutils as crsu
from typing import Optional, List, Tuple, Dict, Union
import mrdja.geometry as geom

def get_np_array_of_three_random_points_from_np_array_of_points(points: np.ndarray, num_points: Optional[int] = None) -> np.ndarray:
    """
    Returns three random points from a list of points.

    :param points: Points.
    :type points: np.ndarray
    :return: Three random points.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import coreransac
        >>> import numpy as np
        >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        >>> points
        array([[0, 0, 0],
               [1, 0, 0],
               [0, 1, 0]])
        >>> random_points = coreransac.get_three_random_points(points)
        >>> random_points
        array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 0]])
    """
    if num_points is None:
        num_points = len(points)
    random_points_indices = np.random.choice(range(num_points), 3, replace=False)
    random_points = points[random_points_indices]
    return random_points

def get_how_many_below_threshold_between_plane_and_points_and_their_indices(points: np.ndarray, plane: np.ndarray, threshold: np.float32) -> Tuple[int, np.ndarray]:
    """
    Computes how many points are below a threshold distance from a plane and returns their count and their indices.

    :param a: A parameter of the plane.
    :type a: np.float32
    :param b: B parameter of the plane.
    :type b: np.float32
    :param c: C parameter of the plane.
    :type c: np.float32
    :param d: D parameter of the plane.
    :type d: np.float32
    :param points_x: X coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: Y coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: Z coordinates of the points.
    :type points_z: np.ndarray
    :param threshold: Maximum distance to the plane.
    :type threshold: np.float32
    :return: Number of points below the threshold distance as their indices.
    :rtype: Tuple[int, np.ndarray]

    :Example:

    ::

        >>> import customransac
        >>> plane = np.array([1,1,1,1], dtype="float32")
        >>> points = [[-1, -1, -1], [0, 0, 0], [1, 1, 1], [2, 2, 2]]
        >>> points_x = np.array([p[0] for p in points], dtype="float32")
        >>> points_y = np.array([p[1] for p in points], dtype="float32")
        >>> points_z = np.array([p[2] for p in points], dtype="float32")
        >>> threshold = 1
        >>> count, indices = customransac.get_how_many_below_threshold_between_plane_and_points_and_indices(plane[0], plane[1], plane[2], plane[3], points_x, points_y, points_z, threshold)
        >>> count
        1
        >>> indices
        array([1])
    """
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]
    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]
    # the distance between the point P with coordinates (xo, yo, zo) and the given plane with equation Ax + By + Cz = D, 
    # is given by |Axo + Byo+ Czo + D|/√(A^2 + B^2 + C^2).
    denominator = math.sqrt(a * a + b * b + c * c)
    optimized_threshold = threshold * denominator  # for not computing the division for each point
    distance = np.abs(points_x * a + points_y * b + points_z * c + d)
    # point_indices = np.where(distance <= optimized_threshold)[0]
    indices_inliers = np.array([index for index, value in enumerate(distance) if value <= optimized_threshold], dtype=np.int64)
    return len(indices_inliers), indices_inliers

def get_pointcloud_from_indices(pcd: o3d.geometry.PointCloud, indices: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Get a point cloud from a list of indices.
    
    :param pcd: Point cloud.
    :type pcd: open3d.geometry.PointCloud
    :param indices: Indices.
    :type indices: np.ndarray
    :return: Point cloud.
    :rtype: open3d.geometry.PointCloud

    :Example:

    ::

        >>> import open3d as o3d
        >>> import customransac
        >>> pcd = o3d.io.read_point_cloud("tests/data/fragment.ply")
        >>> indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> pcd = customransac.get_pointcloud_from_indices(pcd, indices)
        >>> len(pcd.points)
        10
    """
    pcd = pcd.select_by_index(indices)
    return pcd

def get_ransac_iteration_results(points: np.ndarray, num_points: int, threshold: float) -> dict:
    """
    Returns the results of one iteration of the RANSAC algorithm for plane fitting.
    
    :param points: The collection of points to fit the plane to.
    :type points: np.ndarray
    :param num_points: The number of points to randomly select to fit the plane in this iteration.
    :type num_points: int
    :param threshold: The maximum distance from a point to the plane for it to be considered an inlier.
    :type threshold: float
    :return: A dictionary containing the current plane parameters, number of inliers, and their indices.
    :rtype: dict
    """
    current_random_points = get_np_array_of_three_random_points_from_np_array_of_points(points, num_points)
    current_plane = geom.get_plane_from_list_of_three_points(current_random_points.tolist())
    how_many_in_plane, current_point_indices = get_how_many_below_threshold_between_plane_and_points_and_their_indices(points, current_plane, threshold)
    print(num_points, current_random_points, current_plane, how_many_in_plane)
    return {"current_plane": current_plane, "number_inliers": how_many_in_plane, "indices_inliers": current_point_indices}
    

def get_ransac_results(points: np.ndarray, num_points: int, threshold: float, num_iterations: int) -> dict:
    """
    Computes the best plane that fits a collection of points and the indices of the inliers.
    
    :param points_x: X coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: Y coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: Z coordinates of the points.
    :type points_z: np.ndarray
    :param threshold: Maximum distance to the plane.
    :type threshold: np.float32
    :param num_iterations: Number of iterations to compute the best plane.
    :type num_iterations: np.int64
    :param random_points_indices_1: Indices of the points to use as first random point in each iteration.
    :type random_points_indices_1: np.ndarray
    :param random_points_indices_2: Indices of the points to use as second random point in each iteration.
    :type random_points_indices_2: np.ndarray
    :param random_points_indices_3: Indices of the points to use as third random point in each iteration.
    :type random_points_indices_3: np.ndarray
    :return: Best plane parameters and the indices of the points that fit the plane.
    :rtype: Tuple[np.ndarray, np.ndarray]

    :Example:

    ::

        >>> import customransac
        >>> import open3d as o3d
        >>> import numpy as np
        >>> import random
        >>> pcd_filename = "/tmp/Lantegi/Kubic.ply"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> pcd_points = np.asarray(pcd.points, dtype="float32")
        >>> num_points = len(pcd_points)
        >>> pcd_points_x = pcd_points[:, 0]
        >>> pcd_points_y = pcd_points[:, 1]
        >>> pcd_points_z = pcd_points[:, 2]
        >>> threshold = 0.1
        >>> num_iterations = 2
        >>> list_num_points = list(range(num_points))
        >>> random_points_indices_1 = np.array([], dtype="int64")
        >>> random_points_indices_2 = np.array([], dtype="int64")
        >>> random_points_indices_3 = np.array([], dtype="int64")
        >>> for i in range(num_iterations):
        >>>     random_points_indices = random.sample(list_num_points, 3)
        >>>     random_points_indices_1 = np.append(random_points_indices_1, random_points_indices[0])
        >>>     random_points_indices_2 = np.append(random_points_indices_2, random_points_indices[1])
        >>>     random_points_indices_3 = np.append(random_points_indices_3, random_points_indices[2])
        >>> plane_parameters, indices = customransac.get_best_plane_and_inliers(pcd_points_x, pcd_points_y, pcd_points_z, threshold, num_iterations, random_points_indices_1, random_points_indices_2, random_points_indices_3)
        >>> plane_parameters
        array([ 0.0012,  0.0012,  0.0012,  0.0012])
        >>> inlier_cloud = pcd.select_by_index(inliers)
        >>> inlier_cloud.paint_uniform_color([1.0, 0, 0])
        >>> outlier_cloud = pcd.select_by_index(inliers, invert=True)
        >>> o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    """
    best_plane = None
    number_points_in_best_plane = 0
    for _ in range(num_iterations):
        dict_results = get_ransac_iteration_results(points, num_points, threshold)
        current_plane = dict_results["current_plane"]
        how_many_in_plane = dict_results["number_inliers"]
        current_indices_inliers = dict_results["indices_inliers"]
        if how_many_in_plane > number_points_in_best_plane:
            inliers_ratio = how_many_in_plane / num_points
            max_num_iterations = crsu.compute_number_iterations(inliers_ratio, alpha = 0.05)
            print("Current inliers ratio: ", inliers_ratio, " Max num iterations: ", max_num_iterations)
            number_points_in_best_plane = how_many_in_plane
            best_plane = current_plane
            indices_inliers = current_indices_inliers.copy()
    return {"best_plane": best_plane, "number_inliers": number_points_in_best_plane, "indices_inliers": indices_inliers}
