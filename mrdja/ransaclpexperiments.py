import mrdja.ransaclp as ransaclp
import mrdja.pointcloud as pointcloud
import mrdja.ransac.coreransac as coreransac
import mrdja.geometry as geom
import mrdja.drawing as drawing
# import mrdja.ransac.coreransaccuda as coreransaccuda
from typing import Dict, List, Tuple
import numpy as np
import open3d as o3d
import glob
import pickle as pkl

def print_dict_structure(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print(' ' * indent + f'[{key}]')  # Print key as a section header
            print_dict_structure(value, indent + 2)  # Recursively print sub-dictionary
        else:
            print(' ' * indent + key)  # Print key


def get_baseline(filename: str, threshold: float, n_iterations = 100000) -> Dict:
    '''
    Get the open3d results for plane segmentation for a high number of iterations.
    
    :param filename: The path to the file to be processed.
    :type filename: str
    :param threshold: The threshold to be used in the RANSAC algorithm.
    :type threshold: float
    :param n_iterations: The number of iterations to be used in the RANSAC algorithm.
    :type n_iterations: int
    :return: A dictionary with the results.
    :rtype: Dict

    :Example:

    ::

        >>> import mrdja.ransaclpexperiments as experiments
        >>> filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/WC_1/WC_1.ply"
        >>> threshold = 0.02
        >>> n_iterations = 100000
        >>> dict_results = experiments.get_baseline(filename, threshold, n_iterations)
        >>> print(dict_results)
        {'plane_model': array([-0.00485739, -0.00538778,  0.99997369,  0.10760573]), 'number_inliers': 154556}
    
    '''
    pcd = o3d.io.read_point_cloud(filename)
    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                            ransac_n=3,
                                            num_iterations=n_iterations)
    return {"plane_model": plane_model, "number_inliers": len(inliers)}

def get_baseline_S3DIS(database_path: str, threshold: float, n_iterations = 100000) -> Dict:
    '''
    Get the open3d results for plane segmentation in the S3DIS dataset for a high number of iterations.

    :param database_path: The path to the S3DIS dataset.
    :type database_path: str
    :param threshold: The threshold to be used in the RANSAC algorithm.
    :type threshold: float
    :param n_iterations: The number of iterations to be used in the RANSAC algorithm.
    :type n_iterations: int
    :return: A dictionary with the results.
    :rtype: Dict

    :Example:

    ::

        >>> import mrdja.ransaclpexperiments as experiments
        >>> import pickle as pkl 
        >>> database_path = "/home/scpmaotj/Stanford3dDataset_v1.2/"
        >>> threshold = 0.02
        >>> n_iterations = 100000
        >>> dict_results = experiments.get_baseline_S3DIS(database_path, threshold, n_iterations)
        >>> dict_results['/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/office_29/office_29.ply']
        {'plane_model': array([-0.00794638,  0.00484077,  0.99995671, -2.77513434]),
        'number_inliers': 71578}
        >>> # save results to a pickle file
        >>> with open('results_baseline_S3DIS.pkl', 'wb') as f:
        ...     pkl.dump(dict_results, f)


    '''
    ply_files = glob.glob(database_path + "/**/*.ply", recursive=True)
    dict_all_results = {}

    for filename in ply_files:
        current_file_baseline = get_baseline(filename, threshold = threshold, n_iterations = n_iterations)
        dict_all_results[filename] = current_file_baseline

    return dict_all_results

def compute_parameters_ransac_line (line_iterations: int, percentage_chosen_lines: float = 0.2, percentage_chosen_planes: float = 0.05) -> Dict:
    '''
    Compute the parameters for the RANSAC line algorithm from the number of iterations.

    :param line_iterations: The number of iterations to be used in the RANSAC line algorithm.
    :type line_iterations: int
    :return: A tuple with the number of chosen lines, the number of line pairs, the number of chosen planes and the total number of iterations.
    :rtype: Dict

    :Example:

    ::

        >>> import mrdja.ransaclpexperiments as experiments
        >>> line_iterations = 200
        >>> experiments.compute_parameters_ransac_line(line_iterations)
        {'number_chosen_lines': 40,
        'number_lines_pairs': 780,
        'number_chosen_planes': 39,
        'total_iterations': 239}
        >>> experiments.compute_parameters_ransac_line(line_iterations, percentage_chosen_lines = 0.1, percentage_chosen_planes = 0.1)
        {'number_chosen_lines': 20,
        'number_lines_pairs': 190,
        'number_chosen_planes': 19,
        'total_iterations': 219}
    '''
    number_chosen_lines = int(line_iterations * percentage_chosen_lines)
    number_lines_pairs = int (number_chosen_lines * (number_chosen_lines - 1) / 2)
    number_chosen_planes = int(number_lines_pairs * percentage_chosen_planes)
    total_iterations = line_iterations + number_chosen_planes
    return {"number_chosen_lines": number_chosen_lines, "number_lines_pairs": number_lines_pairs, "number_chosen_planes": number_chosen_planes, "total_iterations": total_iterations}

# @profile
def get_data_comparison_ransac_and_ransaclp(filename: str, 
                                            repetitions: int, iterations_list: List[int], threshold: float, 
                                            percentage_chosen_lines: float, percentage_chosen_planes: float, 
                                            cuda: bool = False, verbosity_level: int = 0, 
                                            inherited_verbose_string: str = "",
                                            seed: int = None) -> Dict :
    '''
    Get the data for the comparison between RANSAC and RANSACLP.

    :param filename: The path to the file to be processed.
    :type filename: str
    :param repetitions: The number of repetitions to be used.
    :type repetitions: int
    :param iterations_list: The list of iterations to be used.
    :type iterations_list: List[int]
    :param threshold: The threshold to be used in the RANSAC algorithm.
    :type threshold: float
    :param percentage_chosen_lines: The percentage of chosen lines to be used in the RANSAC line algorithm.
    :type percentage_chosen_lines: float
    :param percentage_chosen_planes: The percentage of chosen planes to be used in the RANSAC line algorithm.
    :type percentage_chosen_planes: float
    :param verbosity_level: The verbosity level to be used.
    :type verbosity_level: int
    :param inherited_verbose_string: The inherited verbose string to be used.
    :type inherited_verbose_string: str
    :param seed: The seed to be used.
    :type seed: int
    :return: A dictionary with the results.
    :rtype: Dict
    '''
    if iterations_list is None:
        raise ValueError("iterations_list cannot be None")
    if len(iterations_list) == 0:
        raise ValueError("iterations_list cannot be empty")
    if repetitions <= 0:
        raise ValueError("repetitions must be greater than 0")
    if percentage_chosen_lines <= 0 or percentage_chosen_lines > 1:
        raise ValueError("percentage_chosen_lines must be greater than 0 and less or equal than 1")
    if percentage_chosen_planes <= 0 or percentage_chosen_planes > 1:
        raise ValueError("percentage_chosen_planes must be greater than 0 and less or equal than 1")
    
    if seed is not None:
        np.random.seed(seed)
    
    dict_all_results = {}
    dict_all_results["filename"] = filename

    pcd = o3d.io.read_point_cloud(filename)
    pcd = pointcloud.pointcloud_sanitize(pcd)
    np_points = np.asarray(pcd.points)

    dict_all_results["number_pcd_points"] = len(np_points)
    dict_all_results["threshold"] = threshold

    # order iterations_list in descending order
    iterations_list.sort(reverse=True)
    max_iterations = iterations_list[0]
    ransaclp_full_data_from_maximum_number_of_iterations = [[]] * repetitions

    for index, num_iterations in enumerate(iterations_list):
        inherited_verbose_string_in_first_loop = f"{inherited_verbose_string} Current max RANSAC iterations {num_iterations} {index+1}/{len(iterations_list)}"
        if verbosity_level > 0:
            print(f"{inherited_verbose_string_in_first_loop} Current number of iterations analyzed: {num_iterations} / {max_iterations}")
        parameters_experiment = compute_parameters_ransac_line(num_iterations, percentage_chosen_lines = percentage_chosen_lines, 
                                                               percentage_chosen_planes = percentage_chosen_planes)
        total_iterations = parameters_experiment["total_iterations"]

        dict_standard_RANSAC_results_list = list()
        dict_line_RANSAC_results_list = list()

        current_repetition = 0
        
        for j in range(repetitions):
            current_repetition = current_repetition + 1
            inherited_verbose_string_in_second_loop = f"{inherited_verbose_string_in_first_loop} Repetition {current_repetition}/{repetitions} "
            # if index == 0:
            ransaclp_best_data, ransaclp_full_data = ransaclp.get_ransaclp_data_from_np_points(np_points, ransac_iterations = num_iterations, 
                                                           threshold = threshold,
                                                           cuda = cuda,
                                                           verbosity_level = verbosity_level, 
                                                           inherited_verbose_string = inherited_verbose_string_in_second_loop,
                                                           seed = None)
            '''
                ransaclp_full_data_from_maximum_number_of_iterations[j] = ransaclp_full_data
            else:
                ransaclp_full_data = ransaclp_full_data_from_maximum_number_of_iterations[j]
                ransaclp_current_iterations_full_data = {}
                ransaclp_current_iterations_results = ransaclp_full_data["ransac_iterations_results"][:num_iterations]
                ransaclp_current_best_iterations_results = max(ransaclp_current_iterations_results, key=lambda x:x["number_inliers"])
                ransaclp_current_iterations_full_data["ransac_iterations_results"] = ransaclp_current_iterations_results
                ransaclp_current_iterations_full_data["ransac_best_iteration_results"] = ransaclp_current_best_iterations_results
                pair_lines_number_inliers = ransaclp.get_lines_and_number_inliers_from_ransac_data_from_file(ransaclp_current_iterations_full_data)
                ordered_list_sse_plane = ransaclp.get_ordered_list_sse_plane(pair_lines_number_inliers, percentage_best = 0.2, verbosity_level=verbosity_level,
                                                        inherited_verbose_string=inherited_verbose_string)
                list_sse_plane_05 = ransaclp.get_n_percentile_from_list_sse_plane(ordered_list_sse_plane, percentile = 5)
                list_good_planes = [sse_plane[1] for sse_plane in list_sse_plane_05]
                if cuda: 
                    ransaclp_best_data = coreransaccuda.get_best_fitting_data_from_list_planes_cuda(np_points, list_good_planes, threshold)
                else:
                    ransaclp_best_data= coreransac.get_best_fitting_data_from_list_planes(np_points, list_good_planes, threshold)
            '''
            ransaclp_number_inliers = ransaclp_best_data["number_inliers"]
            ransaclp_plane = ransaclp_best_data["plane"]
            
            ransac_plane, inliers = pcd.segment_plane(distance_threshold=threshold,
                                                    ransac_n=3,
                                                    num_iterations=total_iterations)
            ransac_number_inliers = len(inliers)

            dict_standard_RANSAC_results = {"number_inliers": ransac_number_inliers, "plane": ransac_plane, "plane_iterations": total_iterations}
            dict_line_RANSAC_results = {"number_inliers": ransaclp_number_inliers, "plane": ransaclp_plane, "line_iterations": num_iterations}
            dict_line_RANSAC_results_list.append(dict_line_RANSAC_results)
            dict_standard_RANSAC_results_list.append(dict_standard_RANSAC_results)

        dict_all_results["standard_RANSAC_" + str(total_iterations)] = dict_standard_RANSAC_results_list
        dict_all_results["line_RANSAC_" + str(total_iterations)] = dict_line_RANSAC_results_list
        # get the mean of n_inliers_maximum of the elements of the list dict_line_RANSAC_results_list
        list_n_inliers_maximum = [int(dict_line_RANSAC_results["number_inliers"]) for dict_line_RANSAC_results in dict_line_RANSAC_results_list]
        mean_n_inliers_maximum = np.mean(list_n_inliers_maximum)
        dict_all_results["mean_number_inliers_line_RANSAC_" + str(total_iterations)] = mean_n_inliers_maximum
        # get the mean of n_inliers_maximum of the elements of the list dict_standard_RANSAC_results_list
        list_n_inliers_maximum = [int(dict_standard_RANSAC_results["number_inliers"]) for dict_standard_RANSAC_results in dict_standard_RANSAC_results_list]
        mean_n_inliers_maximum = np.mean(list_n_inliers_maximum)
        dict_all_results["mean_number_inliers_standard_RANSAC_" + str(total_iterations)] = mean_n_inliers_maximum

    return dict_all_results

def extract_inliers_outliers_plane_pcd_from_pkl_filename(filename_pkl: str, filename_pcd: str, algorithm: str, iteration: int):
    '''
    Extract the inliers and outliers point clouds from a pkl filename.

    :param filename_pkl: The path to the pkl file to be processed.
    :type filename_pkl: str
    :param filename_pcd: The path to the pcd file to be processed.
    :type filename_pcd: str
    :param algorithm: The algorithm to be used.
    :type algorithm: str
    :param iteration: The iteration to be used.
    :type iteration: int
    :return: A tuple with the inliers and outliers point clouds.
    :rtype: Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]

    :Example:

    ::

        >>> import mrdja.ransaclpexperiments as experiments
        >>> import pickle as pkl 
        >>> import open3d as o3d
        >>> filename_pkl = "/home/scpmaotj/Github/mrdja/results_experiments_ransaclp/Open3D/OfficePointClouds/cloud_bin_0.pkl"
        >>> filename_pcd = "/home/scpmaotj/open3d_data/extract/OfficePointClouds/cloud_bin_0.ply"
        >>> algorithm = "line_RANSAC_109"
        >>> iteration = 0
        >>> inliers, outliers = experiments.extract_inliers_outliers_plane_pcd_from_pkl_filename(filename_pkl, filename_pcd, algorithm, iteration)
        >>> # inliers in red
        >>> inliers.paint_uniform_color([1, 0, 0])
        >>> o3d.visualization.draw_geometries([inliers, outliers])
    '''
    with open(filename_pkl, 'rb') as f:
        data = pkl.load(f)
    threshold = data["threshold"]
    data = data[algorithm]
    data = data[iteration]
    plane = data["plane"]
    pcd = o3d.io.read_point_cloud(filename_pcd)
    points = np.asarray(pcd.points)
    how_many, indices = coreransac.get_how_many_below_threshold_between_plane_and_points_and_their_indices(points, plane, threshold) 
    inliers = pcd.select_by_index(indices)
    outliers = pcd.select_by_index(indices, invert=True)
    return inliers, outliers

def get_pointclouds_of_inliers_of_lines_along_with_plane(pcd: o3d.geometry.PointCloud, line1: np.ndarray, line2: np.ndarray, threshold: float) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    '''
    Get the pointclouds of the inliers of the lines that are in indices index_1 and index_2 according to the best number of inliers, 
    along with the equation of the plane that best fits the two lines.

    :param pcd: The point cloud to be processed.
    :type pcd: o3d.geometry.PointCloud
    :param line1: The first line.
    :type line1: np.ndarray
    :param line2: The second line.
    :type line2: np.ndarray
    :param threshold: The threshold to be used.
    :type threshold: float
    :return: A tuple with the inliers and outliers point clouds and the equation of the plane.
    :rtype: Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, np.ndarray]

    :Example:

    ::

        >>> import mrdja.ransaclpexperiments as experiments
        >>> import pickle as pkl
        >>> import open3d as o3d
        >>> filename_pcd = "/home/scpmaotj/open3d_data/extract/OfficePointClouds/cloud_bin_0.ply"
        >>> threshold = 0.02
        >>> iterations = 500
        >>> percentage_chosen_lines = 0.2
        >>> percentage_chosen_planes = 0.05

    '''
    result = dict()
    plane, sse = geom.get_best_plane_from_points_from_two_segments(line1, line2)
    result["plane"] = plane
    np_points = np.asarray(pcd.points)
    how_many, indices1 = coreransac.get_how_many_below_threshold_between_line_and_points_and_their_indices(np_points, line1, threshold) 
    inliers_line1 = pcd.select_by_index(indices1)
    how_many, indices2 = coreransac.get_how_many_below_threshold_between_line_and_points_and_their_indices(np_points, line2, threshold)
    inliers_line2 = pcd.select_by_index(indices2)
    result["inliers_line1"] = inliers_line1
    result["inliers_line2"] = inliers_line2
    how_many, indices = coreransac.get_how_many_below_threshold_between_plane_and_points_and_their_indices(np_points, plane, threshold) 
    inliers_plane = pcd.select_by_index(indices)
    result["inliers_plane"] = inliers_plane
    return result

def show_list_of_inliers_pcd(pcd: o3d.geometry.PointCloud, list_inliers: List[o3d.geometry.PointCloud], 
                             threshold: float = 0.001, color: List[float] = [1, 0, 0], 
                             lines = None, list_colors: List[List[float]] = None):
    '''
    Show the list of inliers point clouds along with the original point cloud. The inliers are painted in "color".
    The outliers is the original point cloud once substrated the inliers.
    '''
    outliers = pcd
    for index, inliers in enumerate(list_inliers):
        outliers = pointcloud.get_pointcloud_after_substracting_point_cloud(outliers, inliers, threshold = threshold)
        if list_colors is not None:
            color = list_colors[index]
        inliers.paint_uniform_color(color)
    if lines is not None:
        aux = list_inliers + [outliers]
        aux.append(lines)
        o3d.visualization.draw_geometries(aux)
    else:
        o3d.visualization.draw_geometries(list_inliers + [outliers])
    # return the pointcloud with everything
    return list_inliers + [outliers]


def get_processing_examples(filename_pcd: str, threshold: float, iterations: int, percentage_chosen_lines: float, percentage_chosen_planes: float, indices: np.ndarray):
    '''
    Get the pointclouds of the inliers of the lines that are in indices index_1 and index_2 according to the best number of inliers, 
    along with the equation of the plane that best fits the two lines.

    :param filename_pcd: The path to the pcd file to be processed.
    :type filename_pcd: str
    :param threshold: The threshold to be used in the RANSAC algorithm.
    :type threshold: float
    :param iterations: The number of iterations to be used in the RANSAC algorithm.
    :type iterations: int
    :param percentage_chosen_lines: The percentage of chosen lines to be used in the RANSAC line algorithm.
    :type percentage_chosen_lines: float
    :param percentage_chosen_planes: The percentage of chosen planes to be used in the RANSAC line algorithm.
    :type percentage_chosen_planes: float
    :return: A tuple with the inliers and outliers point clouds.
    :rtype: Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]

    :Example:

    ::

        >>> import mrdja.ransaclpexperiments as experiments
        >>> import pickle as pkl
        >>> import open3d as o3d
        >>> filename_pcd = "/home/scpmaotj/open3d_data/extract/OfficePointClouds/cloud_bin_10.ply"
        >>> threshold = 0.02
        >>> iterations = 600
        >>> # iterations = 20
        >>> percentage_chosen_lines = 0.2
        >>> percentage_chosen_planes = 0.05
        >>> indices = [0, 10]
        >>> # indices = [0, 1]
        >>> indices = [50, 51]
        >>> indices = [51, 52]
        >>> indices = [53, 54]
        >>> indices = [50, 54]
        >>> indices = [0, 1]
        >>> results = experiments.get_processing_examples(filename_pcd, threshold, iterations, percentage_chosen_lines, percentage_chosen_planes, indices = indices)
        >>> # inliers_o3d = results["inliers_o3d"]
        >>> # outliers_o3d = results["outliers_o3d"]
        >>> # inliers in red
        >>> # inliers_o3d.paint_uniform_color([1, 0, 0])
        >>> # o3d.visualization.draw_geometries([inliers_o3d, outliers_o3d])

    '''

    results = dict()
    pcd = o3d.io.read_point_cloud(filename_pcd)
    np_points = np.asarray(pcd.points)

    index_1 = indices[0]
    index_2 = indices[1]

    verbosity_level = 0
    inherited_verbose_string = ""
    seed = 42
    ransac_iterator = coreransac.get_ransac_line_iteration_results
    ransac_data = ransaclp.get_ransac_data_from_np_points(np_points, ransac_iterator = ransac_iterator,
                                                    ransac_iterations = iterations,
                                                    threshold = threshold,
                                                    verbosity_level = verbosity_level, 
                                                    inherited_verbose_string=inherited_verbose_string, seed = seed)
    ransac_iterations_results = ransac_data["ransac_iterations_results"]
    # ransac_iterations_results is a list of dictionaries;m order it by "number_inliers"
    ransac_iterations_results.sort(key=lambda x:x["number_inliers"], reverse=True)
    line1 = ransac_iterations_results[index_1]["current_line"]
    line2 = ransac_iterations_results[index_2]["current_line"]
    d = get_pointclouds_of_inliers_of_lines_along_with_plane(pcd, line1, line2, threshold = threshold)
    inliers_line_1 = d["inliers_line1"]
    inliers_line_2 = d["inliers_line2"]
    list_inliers = [inliers_line_1, inliers_line_2]
    final_pointcloud = show_list_of_inliers_pcd(pcd, list_inliers, threshold=0.001)
    plane = d["plane"]
    inliers_plane = d["inliers_plane"]
    list_inliers = [inliers_plane]
    final_pointcloud = show_list_of_inliers_pcd(pcd, list_inliers, color = [0, 1, 0], threshold=0.001)
    # compute centroid of pcd
    centroid = np.mean(np_points, axis=0)
    lines = drawing.draw_plane_as_lines_open3d(*plane, external_point=centroid, size=1.5, grid_density=40, line_color=[0, 0, 1])
    plane_plus_lines = show_list_of_inliers_pcd(inliers_plane, [inliers_line_1, inliers_line_2], threshold=0.001)[-1]
    list_inliers = [inliers_line_1, inliers_line_2, plane_plus_lines]
    list_colors = [[1, 0, 0], [1, 0, 0], [0, 1, 0]]
    final_pointcloud = show_list_of_inliers_pcd(pcd, list_inliers, list_colors = list_colors, lines = lines)
    '''
    for index, ransac_iteration_result in enumerate(ransac_iterations_results):
        inliers_line_1 = pcd.select_by_index(ransac_iterations_results[0]["indices_inliers"])
        inliers_line_2 = pcd.select_by_index(ransac_iteration_result["indices_inliers"])
        # show inliers_line_1 and inliers_line_2 in red and outliers in original color
        inliers_line_1.paint_uniform_color([1, 0, 0])
        inliers_line_2.paint_uniform_color([1, 0, 0])
        outliers = pcd.select_by_index(ransac_iterations_results[0]["indices_inliers"], invert=True)
        outliers = pointcloud.get_pointcloud_after_substracting_point_cloud(outliers, inliers_line_2, threshold = 0.0001)
        print(index)
        o3d.visualization.draw_geometries([inliers_line_1, inliers_line_2, outliers])
    '''

    '''
    pair_lines_number_inliers = ransaclp.get_lines_and_number_inliers_from_ransac_data_from_file(ransac_data)
    ordered_list_sse_plane = ransaclp.get_ordered_list_sse_plane(pair_lines_number_inliers, percentage_best = 0.2, verbosity_level=verbosity_level,
                                                        inherited_verbose_string=inherited_verbose_string)
    list_sse_plane_05 = ransaclp.get_n_percentile_from_list_sse_plane(ordered_list_sse_plane, percentile = 5)
    list_good_planes = [sse_plane[1] for sse_plane in list_sse_plane_05]
    results_from_best_plane = coreransac.get_best_fitting_data_from_list_planes(np_points, list_good_planes, threshold)
    best_plane = results_from_best_plane["plane"]
    how_many, indices = coreransac.get_how_many_below_threshold_between_plane_and_points_and_their_indices(np_points, best_plane, threshold) 
    inliers_plane = pcd.select_by_index(indices)
    inliers_plane.paint_uniform_color([1, 0, 0])
    outliers = pcd.select_by_index(indices, invert=True)
    o3d.visualization.draw_geometries([inliers_plane, outliers])
    # segment plane with open3d
    parameters = compute_parameters_ransac_line(iterations, percentage_chosen_lines = 0.2, percentage_chosen_planes = 0.05)
    total_iterations = parameters["total_iterations"]

    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                            ransac_n=3,
                                            num_iterations=total_iterations)
    inliers_o3d = pcd.select_by_index(inliers)
    outliers_o3d = pcd.select_by_index(inliers, invert=True)
    # paint inliers in red
    inliers_o3d.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([inliers_o3d, outliers_o3d])
    return results_from_best_plane, ransac_data
    '''





    '''
    ransaclp_best_data, ransaclp_full_data = ransaclp.get_ransaclp_data_from_np_points(np_points, ransac_iterations = iterations, 
                                                           threshold = threshold,
                                                           cuda = False,
                                                           verbosity_level = 0, 
                                                           inherited_verbose_string = "",
                                                           seed = None)
    print_dict_structure(ransaclp_best_data)
    # get the inliers and outliers from these data
    inliers = ransaclp_best_data["inliers"]
    outliers = ransaclp_best_data["outliers"]
    # use compute_parameters_ransac_line to get the number of iterations
    parameters_experiment = compute_parameters_ransac_line(iterations, percentage_chosen_lines = percentage_chosen_lines, 
                                                               percentage_chosen_planes = percentage_chosen_planes)
    total_iterations = parameters_experiment["total_iterations"]
        # get the best plane with o3d
    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                            ransac_n=3,
                                            num_iterations=total_iterations)
    inliers_o3d = pcd.select_by_index(inliers)
    outliers_o3d = pcd.select_by_index(inliers, invert=True)
    # paint inliers in red
    # inliers_o3d.paint_uniform_color([1, 0, 0])
    results["inliers_o3d"] = inliers_o3d
    results["outliers_o3d"] = outliers_o3d
    return results
    '''
    