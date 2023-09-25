from typing import Dict
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
