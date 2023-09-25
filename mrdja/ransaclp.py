import mrdja.ransac.coreransac as coreransac
import mrdja.pointcloud as pointcloud
import mrdja.geometry as geom
import mrdja.drawing as drawing
import open3d as o3d
import numpy as np
import random
from typing import Callable, List, Tuple, Union, Dict

coreransac.get_ransac_line_iteration_results

def get_ransac_data_from_filename(filename: str, ransac_iterator: Callable, ransac_iterations: int = 100, threshold: float = 0.1, audit_cloud: bool = False, seed: int = None) -> Dict:
    '''
    Gets the ransac data from a file. 
    
    The ransac data is a dictionary with the following keys:
    - filename: the filename
    - audit_before_sanitizing: the audit of the point cloud before sanitizing; it is a dictionary with the following keys:
        - number_pcd_points: the number of points in the point cloud
        - has_normals: whether the point cloud has normals
        - has_colors: whether the point cloud has colors
        - is_empty: whether the point cloud is empty
        - max_x: the maximum x coordinate
        - min_x: the minimum x coordinate
        - max_y: the maximum y coordinate
        - min_y: the minimum y coordinate
        - max_z: the maximum z coordinate
        - min_z: the minimum z coordinate
        - all_points_finite: whether all the points are finite
        - all_points_unique: whether all the points are unique
    - audit_after_sanitizing: the audit of the point cloud after sanitizing
    - number_pcd_points: the number of points in the point cloud
    - ransac_iterations_results: the results of the ransac iterations. It is a list, where each element is a dictionary with the following keys:
        - current_random_points: the current random points chosen to estimate the line
        - current_line: the current line estimated from the random points. Right now it is a numpy array with two points, the same as the current_random_points
        - threshold: the maximum distance from the line to consider a point as an inlier
        - number_inliers: the number of inliers
        - indices_inliers: the indices of the inliers in the point cloud
    - ransac_best_iteration_results: the best iteration results

    :param filename: The filename.
    :type filename: str
    :param ransac_iterator: The ransac iterator.
    :type ransac_iterator: Callable, function that takes the point cloud data and returns the ransac data.
    :param ransac_iterations: The number of ransac iterations.
    :type ransac_iterations: int
    :param threshold: The threshold.
    :type threshold: float
    :param seed: The seed.
    :type seed: int
    :return: The ransac data.
    :rtype: dict

    :Example:

    ::

        >>> import open3d as o3d
        >>> import mrdja.ransaclp as ransaclp
        >>> import mrdja.ransac.coreransac as coreransac
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> ransac_iterator = coreransac.get_ransac_line_iteration_results
        >>> ransac_iterations = 200
        >>> threshold = 0.02
        >>> seed = 42
        >>> ransac_data = ransaclp.get_ransac_data_from_filename(office_filename, ransac_iterator = ransac_iterator, 
                                                           ransac_iterations = ransac_iterations, 
                                                           threshold = threshold, audit_cloud=True, seed = seed)
        >>> ransac_data["filename"]
        '/home/user/open3d_data/extract/OfficePointClouds/cloud_bin_0.ply'
        >>> ransac_data["audit_before_sanitizing"]
        {'number_pcd_points': 276871,
        'has_normals': True,
        'has_colors': True,
        'is_empty': False,
        'max_x': 3.5121824741363525,
        'min_x': 0.00390625,
        'max_y': 2.80859375,
        'min_y': 0.47265625,
        'max_z': 2.7512423992156982,
        'min_z': 0.94921875,
        'all_points_finite': True,
        'all_points_unique': True}
        >>> ransac_data["audit_after_sanitizing"]
        {'number_pcd_points': 276871,
        'has_normals': True,
        'has_colors': True,
        'is_empty': False,
        'max_x': 3.5121824741363525,
        'min_x': 0.00390625,
        'max_y': 2.80859375,
        'min_y': 0.47265625,
        'max_z': 2.7512423992156982,
        'min_z': 0.94921875,
        'all_points_finite': True,
        'all_points_unique': True}
        >>> ransac_data["number_pcd_points"]
        276871
        >>> iterations_results = ransac_data["ransac_iterations_results"]
        >>> len(iterations_results)
        200
        >>> first_iteration_results = iterations_results[0]
        >>> first_iteration_results["current_random_points"]
        array([[1.88671875, 1.96484375, 1.91060746],
        [3.18359375, 1.89562058, 2.45703125]])
        >>> first_iteration_results["current_line"]
        array([[1.88671875, 1.96484375, 1.91060746],
        [3.18359375, 1.89562058, 2.45703125]])
        >>> first_iteration_results["threshold"]
        0.02
        >>> first_iteration_results["number_inliers"]
        203
        >>> first_iteration_results["indices_inliers"][:5]
        array([ 58884,  59966,  60516,  61070, 138037])
        >>> best_iteration_results = ransac_data["ransac_best_iteration_results"]
        >>> best_iteration_results["current_random_points"]
        array([[1.85546875, 2.67396379, 2.08203125],
        [0.85546875, 2.57869077, 2.52734375]])
        >>> best_iteration_results["current_line"]
        array([[1.85546875, 2.67396379, 2.08203125],
        [0.85546875, 2.57869077, 2.52734375]])
        >>> best_iteration_results["threshold"]
        0.02
        >>> best_iteration_results["number_inliers"]
        2128
        >>> best_iteration_results["indices_inliers"][:5]
        array([24335, 25743, 25746, 25897, 26405])
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
        dict_iteration_results = ransac_iterator(np_points, threshold, number_pcd_points)
        if dict_iteration_results["number_inliers"] > max_number_inliers:
            max_number_inliers = dict_iteration_results["number_inliers"]
            best_iteration_results = dict_iteration_results
        dict_full_results["ransac_iterations_results"].append(dict_iteration_results)
    dict_full_results["ransac_best_iteration_results"] = best_iteration_results
    return dict_full_results

# create a function that returns all the current_line along with their number_inliers

def get_lines_and_number_inliers_from_ransac_data_from_file(ransac_data_from_file):
    pair_lines_number_inliers = []
    for dict_iteration_results in ransac_data_from_file["ransac_iterations_results"]:
        pair_lines_number_inliers.append((dict_iteration_results["current_line"], dict_iteration_results["number_inliers"]))
    return pair_lines_number_inliers

# order the pairs by number_inliers

def get_lines_and_number_inliers_ordered_by_number_inliers(ransac_data: Dict) -> List[Tuple[np.ndarray, int]]:
    '''
    Gets the lines and number of inliers ordered by number of inliers from ransac data previously extracted.

    :param ransac_data: The ransac data.
    :type ransac_data: Dict
    :return: The list of pairs of lines and number of inliers.
    :rtype: List[Tuple[np.ndarray, int]]

    :Example:

    ::

        >>> import open3d as o3d
        >>> import mrdja.ransaclp as ransaclp
        >>> import mrdja.ransac.coreransac as coreransac
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> ransac_iterator = coreransac.get_ransac_line_iteration_results
        >>> ransac_iterations = 200
        >>> threshold = 0.02
        >>> seed = 42
        >>> ransac_data = ransaclp.get_ransac_data_from_filename(office_filename, ransac_iterator = ransac_iterator,
                                                                 ransac_iterations = ransac_iterations,
                                                                 threshold = threshold, audit_cloud=True, seed = seed)
        >>> pair_lines_number_inliers_ordered = ransaclp.get_lines_and_number_inliers_ordered_by_number_inliers(ransac_data)
        >>> pair_lines_number_inliers_ordered[:5]
        [(array([[1.85546875, 2.67396379, 2.08203125],
                [0.85546875, 2.57869077, 2.52734375]]),
        2128),
        (array([[0.19140625, 2.10900044, 2.62890625],
                [1.03637648, 1.11328125, 2.43359375]]),
        1988),
        (array([[3.12890625, 2.17578125, 2.52116251],
                [1.27803993, 0.67578125, 2.33203125]]),
        1960),
        (array([[0.31464097, 0.90234375, 2.46484375],
                [2.99609375, 1.31640625, 2.34281754]]),
        1953),
        (array([[1.06640625, 0.86328125, 2.37622762],
                [1.03515625, 1.08984375, 2.42773271]]),
        1925)]


    '''
    pair_lines_number_inliers = get_lines_and_number_inliers_from_ransac_data_from_file(ransac_data)
    pair_lines_number_inliers_ordered = sorted(pair_lines_number_inliers, key=lambda pair_line_number_inliers: pair_line_number_inliers[1], reverse=True)
    return pair_lines_number_inliers_ordered

def get_ordered_list_sse_plane(pair_lines_number_inliers:List[Tuple[np.ndarray, int]], number_best: int = None, percentage_best: float = 0.2, already_ordered: bool = False):
    '''
    Gets a list of the sse (sum of squared errors) of the planes built from the best pairs of lines, along with the planes. 
    The best pairs of lines are the ones with the highest number of inliers. 
    The number of planes returned is either the number_best or the percentage_best of the total number of planes.

    :param pair_lines_number_inliers: The list of pairs of lines and number of inliers.
    :type pair_lines_number_inliers: List[Tuple[np.ndarray, int]]
    :param number_best: The number of best planes to consider.
    :type number_best: int
    :param percentage_best: The percentage of best planes to consider.
    :type percentage_best: float
    :param already_ordered: Whether the pairs are already ordered by number of inliers.
    :type already_ordered: bool
    :return: The list of sse and planes ordered by sse.
    :rtype: List[Tuple[float, np.ndarray]]

    :Example:

    ::

        >>> import open3d as o3d
        >>> import mrdja.ransaclp as ransaclp
        >>> import mrdja.ransac.coreransac as coreransac
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> ransac_iterator = coreransac.get_ransac_line_iteration_results
        >>> ransac_iterations = 200
        >>> threshold = 0.02
        >>> seed = 42
        >>> ransac_data = ransaclp.get_ransac_data_from_filename(office_filename, ransac_iterator = ransac_iterator,
                                                                    ransac_iterations = ransac_iterations,
                                                                    threshold = threshold, audit_cloud=True, seed = seed)
        >>> pair_lines_number_inliers = ransaclp.get_lines_and_number_inliers_from_ransac_data_from_file(ransac_data)
        >>> list_sse_plane = ransaclp.get_ordered_list_sse_plane(pair_lines_number_inliers, percentage_best = 0.2)
        >>> list_sse_plane[:5]
        [(3.133004699296308e-09,
        array([ 0.02115254,  0.23488194, -0.97179373,  2.07212126])),
        (4.072922344774246e-09,
        array([-0.0583794 ,  0.16947913, -0.98380317,  2.274669  ])),
        (5.597729648076989e-09,
        array([ 0.04009822, -0.15939724,  0.98639984, -2.26464593])),
        (1.6036620702538505e-08,
        array([ 0.04939934, -0.14154284,  0.98869881, -2.30467834])),
        (1.7597194644519755e-08,
        array([-0.02745968,  0.98751003,  0.15514477, -2.9031455 ]))]
    '''
    if number_best is None:
        number_best = int(len(pair_lines_number_inliers) * percentage_best)
    if not already_ordered:
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
    # order the list by sse
    list_sse_plane = sorted(list_sse_plane, key=lambda x: x[0])
    return list_sse_plane

def get_n_percentile_from_list_sse_plane(list_sse_plane: List[Tuple[float, np.ndarray]], percentile: int = 5) -> List[Tuple[float, np.ndarray]]:
    '''
    Gets the list of elements in which sse is below the n percentile from a list of sse and planes.
    The function looks for the best sse values, that are the lowest ones.

    :param list_sse_plane: The list of sse and planes.
    :type list_sse_plane: List[Tuple[float, np.ndarray]]
    :param percentile: The percentile.
    :type percentile: int
    :return: The n percentile.
    :rtype: List[Tuple[float, np.ndarray]]

    :Example:

    ::

        >>> import open3d as o3d
        >>> import mrdja.ransaclp as ransaclp
        >>> import mrdja.ransac.coreransac as coreransac
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> ransac_iterator = coreransac.get_ransac_line_iteration_results
        >>> ransac_iterations = 200
        >>> threshold = 0.02
        >>> seed = 42
        >>> ransac_data = ransaclp.get_ransac_data_from_filename(office_filename, ransac_iterator = ransac_iterator,
                                                                    ransac_iterations = ransac_iterations,
                                                                    threshold = threshold, audit_cloud=True, seed = seed)
        >>> pair_lines_number_inliers = ransaclp.get_lines_and_number_inliers_from_ransac_data_from_file(ransac_data)
        >>> list_sse_plane = ransaclp.get_ordered_list_sse_plane(pair_lines_number_inliers, percentage_best = 0.2)
        >>> len(list_sse_plane)
        780
        >>> list_sse_plane[:5]
        [(3.133004699296308e-09,
        array([ 0.02115254,  0.23488194, -0.97179373,  2.07212126])),
        (4.072922344774246e-09,
        array([-0.0583794 ,  0.16947913, -0.98380317,  2.274669  ])),
        (5.597729648076989e-09,
        array([ 0.04009822, -0.15939724,  0.98639984, -2.26464593])),
        (1.6036620702538505e-08,
        array([ 0.04939934, -0.14154284,  0.98869881, -2.30467834])),
        (1.7597194644519755e-08,
        array([-0.02745968,  0.98751003,  0.15514477, -2.9031455 ]))]
        >>> list_sse_plane_05 = ransaclp.get_n_percentile_from_list_sse_plane(list_sse_plane, percentile = 5)
        >>> len(list_sse_plane_05)
        39
        >>> list_sse_plane_05[:5]
        [(3.133004699296308e-09,
        array([ 0.02115254,  0.23488194, -0.97179373,  2.07212126])),
        (4.072922344774246e-09,
        array([-0.0583794 ,  0.16947913, -0.98380317,  2.274669  ])),
        (5.597729648076989e-09,
        array([ 0.04009822, -0.15939724,  0.98639984, -2.26464593])),
        (1.6036620702538505e-08,
        array([ 0.04939934, -0.14154284,  0.98869881, -2.30467834])),
        (1.7597194644519755e-08,
        array([-0.02745968,  0.98751003,  0.15514477, -2.9031455 ]))]

    '''
    list_sse, list_plane = zip(*list_sse_plane)  # Unpack the list of tuples
    percentile_threshold = np.percentile(list_sse, percentile)
    filtered_pairs = [(sse, plane) for sse, plane in zip(list_sse, list_plane) if sse <= percentile_threshold]
    return filtered_pairs

def get_ransaclp_data_from_filename(filename: str, ransac_iterations: int = 100, threshold: float = 0.1, audit_cloud: bool = False, seed: int = None) -> Dict:
    '''
    Gets the ransaclp data from a file.
    
    :param filename: The filename.
    :type filename: str
    :param ransac_iterations: The number of ransac iterations.
    :type ransac_iterations: int
    :param threshold: The threshold.
    :type threshold: float
    :param seed: The seed.
    :type seed: int
    :return: The ransaclp data.
    :rtype: dict

    :Example:

    ::

        >>> import open3d as o3d
        >>> import mrdja.ransaclp as ransaclp
        >>> import mrdja.ransac.coreransac as coreransac
        >>> office_dataset = o3d.data.OfficePointClouds()
        >>> office_filename = office_dataset.paths[0]
        >>> ransac_iterations = 200
        >>> threshold = 0.02
        >>> seed = 42
        >>> ransaclp_data = ransaclp.get_ransaclp_data_from_filename(office_filename, 
                                                                    ransac_iterations = ransac_iterations, 
                                                                    threshold = threshold, audit_cloud=True, seed = seed)
        >>> ransaclp_data
        {'plane': array([-0.07208637,  0.20357587, -0.97640177,  2.25114528]),
        'number_inliers': 83788,
        'indices_inliers': array([     7,      8,     10, ..., 248476, 248477, 248478])}
        >>> pcd = o3d.io.read_point_cloud(office_filename)
        >>> number_inliers = ransaclp_data["number_inliers"]
        >>> indices_inliers = ransaclp_data["indices_inliers"]
        >>> inlier_cloud = pcd.select_by_index(indices_inliers)
        >>> inlier_cloud.paint_uniform_color([1, 0, 0])
        >>> o3d.visualization.draw_geometries([pcd, inlier_cloud], window_name="RANSACLP Inliers:  " + str(number_inliers))

    |ransaclp_get_ransaclp_data_from_filename_example|

    .. |ransaclp_get_ransaclp_data_from_filename_example| image:: ../../_static/images/ransaclp_get_ransaclp_data_from_filename_example.png

    '''
    if seed is None:
        seed = random.randint(0, 1000000)
    np.random.seed(seed)
    ransac_iterator = coreransac.get_ransac_line_iteration_results
    ransac_data = get_ransac_data_from_filename(filename, ransac_iterator = ransac_iterator,
                                                ransac_iterations = ransac_iterations,
                                                threshold = threshold, audit_cloud=True, seed = seed)
    pcd = o3d.io.read_point_cloud(ransac_data["filename"])
    np_points = np.asarray(pcd.points)

    pair_lines_number_inliers = get_lines_and_number_inliers_from_ransac_data_from_file(ransac_data)
    ordered_list_sse_plane = get_ordered_list_sse_plane(pair_lines_number_inliers, percentage_best = 0.2)
    list_sse_plane_05 = get_n_percentile_from_list_sse_plane(ordered_list_sse_plane, percentile = 5)
    list_good_planes = [sse_plane[1] for sse_plane in list_sse_plane_05]
    results_from_best_plane = coreransac.get_best_fitting_data_from_list_planes(np_points, list_good_planes, threshold)
    return results_from_best_plane