import mrdja.ransac.coreransac as coreransac
import mrdja.pointcloud as pointcloud
import mrdja.geometry as geom
import mrdja.drawing as drawing
import open3d as o3d
import numpy as np
import random
from typing import Callable, List, Tuple, Union, Dict

coreransac.get_ransac_line_iteration_results

def get_RANSAC_data_from_file(filename: str, RANSAC_iterator: Callable, ransac_iterations: int = 100, threshold: float = 0.1, audit_cloud: bool = False, seed: int = None) -> dict:
    '''
    Gets the RANSAC data from a file.

    :param filename: The filename.
    :type filename: str
    :param RANSAC_iterator: The RANSAC iterator.
    :type RANSAC_iterator: Callable, function that takes the point cloud data and returns the RANSAC data.
    :param ransac_iterations: The number of RANSAC iterations.
    :type ransac_iterations: int
    :param threshold: The threshold.
    :type threshold: float
    :param seed: The seed.
    :type seed: int
    :return: The RANSAC data.
    :rtype: dict

    :Example:

    ::
    '''
    if seed is None:
        seed = random.randint(0, 1000000)
    np.random.seed(seed)

    dict_full_results = {}
    dict_full_results["filename"] = filename

    pcd = o3d.io.read_point_cloud(filename)
    if audit_cloud:
        audit_before_sanitizing = pointcloud.pointcloud_audit(pcd)
        dict_full_results["audit_before_sanitizing"] = audit_before_sanitizing
        pcd = pointcloud.pointcloud_sanitize(pcd)
        audit_after_sanitizing = pointcloud.pointcloud_audit(pcd)
        dict_full_results["audit_after_sanitizing"] = audit_after_sanitizing
    
    number_pcd_points = len(pcd.points)
    dict_full_results["number_pcd_points"] = number_pcd_points
    np_points = np.asarray(pcd.points)

    dict_full_results["ransac_iterations_results"] = []

    max_number_inliers = 0
    best_iteration_results = None
    for i in range(ransac_iterations):
        print("Iteration", i)
        dict_iteration_results = RANSAC_iterator(np_points, threshold, number_pcd_points)
        if dict_iteration_results["number_inliers"] > max_number_inliers:
            max_number_inliers = dict_iteration_results["number_inliers"]
            best_iteration_results = dict_iteration_results
        dict_full_results["ransac_iterations_results"].append(dict_iteration_results)
    dict_full_results["ransac_best_iteration_results"] = best_iteration_results
    return dict_full_results

# create a function that returns all the current_line along with their number_inliers

def get_lines_and_number_inliers_from_RANSAC_data_from_file(RANSAC_data_from_file):
    pair_lines_number_inliers = []
    for dict_iteration_results in RANSAC_data_from_file["ransac_iterations_results"]:
        pair_lines_number_inliers.append((dict_iteration_results["current_line"], dict_iteration_results["number_inliers"]))
    return pair_lines_number_inliers

# order the pairs by number_inliers

def get_lines_and_number_inliers_ordered_by_number_inliers(RANSAC_data_from_file):
    pair_lines_number_inliers = get_lines_and_number_inliers_from_RANSAC_data_from_file(RANSAC_data_from_file)
    pair_lines_number_inliers_ordered = sorted(pair_lines_number_inliers, key=lambda pair_line_number_inliers: pair_line_number_inliers[1], reverse=True)
    return pair_lines_number_inliers_ordered

def get_list_sse_plane(pair_lines_number_inliers:List[Tuple[np.ndarray, int]], number_best: int = None, percentage_best: float = 0.2, ordered: bool = False):
    '''
    Gets a list of the sse (sum of squared errors) of the best planes.

    :param pair_lines_number_inliers: The list of pairs of lines and number of inliers.
    :type pair_lines_number_inliers: List[Tuple[np.ndarray, int]]
    :param number_best: The number of best planes to consider.
    :type number_best: int
    :param percentage_best: The percentage of best planes to consider.
    :type percentage_best: float
    :param ordered: Whether the pairs are ordered by number of inliers.
    :type ordered: bool
    :return: The list of sse and planes.
    :rtype: List[Tuple[float, np.ndarray]]

    '''
    if number_best is None:
        number_best = int(len(pair_lines_number_inliers) * percentage_best)
    if not ordered:
        pair_lines_number_inliers = sorted(pair_lines_number_inliers, key=lambda x: x[1], reverse=True)
    list_sse_plane = []
    current_iteration = 0
    for i in range(number_best):
        for j in range(i+1, number_best):
            # compute the current number of iterations
            current_iteration += 1
            print("Iteration", current_iteration)
            line_1 = pair_lines_number_inliers[i][0]
            line_2 = pair_lines_number_inliers[j][0]
            plane, error = geom.get_best_plane_from_points_from_two_segments(line_1, line_2)
            list_sse_plane.append((error, plane))
    return list_sse_plane