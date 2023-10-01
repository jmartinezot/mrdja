import mrdja.ransaclp as ransaclp
import mrdja.pointcloud as pointcloud
from typing import Dict, List
import numpy as np
import open3d as o3d
import glob
import os

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

def get_data_comparison_ransac_and_ransaclp(filename: str, repetitions: int, iterations_list: List[int], threshold: float, 
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

    # get the maximum number of iterations
    max_iterations = max(iterations_list)
    filename_only_file = os.path.basename(filename)
    inherited_verbose_string = inherited_verbose_string + filename_only_file + " "

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
            ransaclp_data_from_file = ransaclp.get_ransaclp_data_from_np_points(np_points, ransac_iterations = num_iterations, 
                                                           threshold = threshold,
                                                           cuda = cuda,
                                                           verbosity_level = verbosity_level, 
                                                           inherited_verbose_string = inherited_verbose_string_in_second_loop,
                                                           seed = None)
            ransaclp_number_inliers = ransaclp_data_from_file["number_inliers"]
            ransaclp_plane = ransaclp_data_from_file["plane"]
            
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
