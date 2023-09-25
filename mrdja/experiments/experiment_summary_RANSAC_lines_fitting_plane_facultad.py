import mrdja.ransac.coreransac as coreransac
import mrdja.ransaclpexperiments as experiments
import open3d as o3d
import numpy as np
import random
import pickle as pkl

'''
Line-guided plane detection in pointclouds.
Tomar, por ejemplo, 200 rectas y luego los 40 mejores por parejas. Eso son 780 parejas. Tomar el 5% mejor según el error de ajuste del plano.
Eso son 39 parejas. En total 239 veces que hay que mirar los inliers mirando todos los puntos.
Calcular el mejor plano, tomando RANSAC de open3d con 100.000 iteraciones.
Hacer varios experimentos, tanto de RANSAC con 239 o más iteraciones, como de open3d RANSAC con otros números de iteraciones.
'''

# The files in the Stanford database are like this:
# {'number_pcd_points': 1136617, 'has_normals': False, 
# 'has_colors': True, 'is_empty': False, 'max_x': -15.207, 
# 'min_x': -20.542, 'max_y': 41.283, 'min_y': 36.802, 
# 'max_z': 3.206, 'min_z': 0.02, 'all_points_finite': True, 
# 'all_points_unique': False}
# the measures are in meters
# dict_results = pointcloud_audit(pcd)
# print(dict_results)

import open3d as o3d
import mrdja.ransaclp as ransaclp
import mrdja.ransac.coreransac as coreransac
import numpy as np

# database_path = "/home/scpmaotj/Stanford3dDataset_v1.2/"
# get a list with all the ply files under database_path
import glob

# database_path = "/home/bee/Lantegi_dataset/Stanford3dDataset_v1.2/"
database_path = "/home/scpmaotj/Stanford3dDataset_v1.2/"
ply_files = glob.glob(database_path + "/**/*.ply", recursive=True)

threshold = 0.02
total_number_files = len(ply_files)

np.random.seed(42)

for index , filename in enumerate(ply_files):

    if index == 3: 
        break

    print(f"filename {index} of {total_number_files}: {filename}")

    dict_all_results = {}
    dict_all_results["filename"] = filename

    pcd = o3d.io.read_point_cloud(filename)
    np_points = np.asarray(pcd.points)

    for num_iterations in [100, 200, 300, 400, 500, 600]:
        parameters_experiment = experiments.compute_parameters_ransac_line(num_iterations, percentage_chosen_lines = 0.2, percentage_chosen_planes = 0.05)
        number_chosen_lines = parameters_experiment["number_chosen_lines"]
        number_lines_pairs = parameters_experiment["number_lines_pairs"]
        number_chosen_planes = parameters_experiment["number_chosen_planes"]
        total_iterations = parameters_experiment["total_iterations"]
        # filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/conferenceRoom_1/conferenceRoom_1.ply"
        # filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/office_1/office_1.ply"

        dict_standard_RANSAC_results_list = list()
        dict_line_RANSAC_results_list = list()

        for j in range(10):
            ransaclp_data_from_file = ransaclp.get_ransaclp_data_from_filename(filename, ransac_iterations = num_iterations, 
                                                           threshold = threshold, audit_cloud=True)
            ransaclp_number_inliers = ransaclp_data_from_file["number_inliers"]
            ransaclp_plane = ransaclp_data_from_file["plane"]
            
            ransac_plane, inliers = pcd.segment_plane(distance_threshold=threshold,
                                                    ransac_n=3,
                                                    num_iterations=total_iterations)
            ransac_number_inliers = len(inliers)

            dict_standard_RANSAC_results = {"n_points_pcd": len(np_points), "n_inliers_maximum": ransac_number_inliers, "best_plane": ransac_plane, "threshold": threshold}
            dict_line_RANSAC_results = {"n_points_pcd": len(np_points), "n_inliers_maximum": ransaclp_number_inliers, "best_plane": ransaclp_plane, "threshold": threshold}
            dict_line_RANSAC_results_list.append(dict_line_RANSAC_results)
            dict_standard_RANSAC_results_list.append(dict_standard_RANSAC_results)

        dict_all_results["standard_RANSAC_" + str(total_iterations)] = dict_standard_RANSAC_results_list
        dict_all_results["line_RANSAC_" + str(total_iterations)] = dict_line_RANSAC_results_list
        # get the mean of n_inliers_maximum of the elements of the list dict_line_RANSAC_results_list
        list_n_inliers_maximum = [int(dict_line_RANSAC_results["n_inliers_maximum"]) for dict_line_RANSAC_results in dict_line_RANSAC_results_list]
        mean_n_inliers_maximum = np.mean(list_n_inliers_maximum)
        dict_all_results["mean_n_inliers_maximum_line_RANSAC_" + str(total_iterations)] = mean_n_inliers_maximum
        # get the mean of n_inliers_maximum of the elements of the list dict_standard_RANSAC_results_list
        list_n_inliers_maximum = [int(dict_standard_RANSAC_results["n_inliers_maximum"]) for dict_standard_RANSAC_results in dict_standard_RANSAC_results_list]
        mean_n_inliers_maximum = np.mean(list_n_inliers_maximum)
        dict_all_results["mean_n_inliers_maximum_standard_RANSAC_" + str(total_iterations)] = mean_n_inliers_maximum

    # save the results as a pickle file in the same folder as the filename file; to do so, just change the extension of the file to pkl
    filename_pkl = filename.replace(".ply", ".pkl")
    with open(filename_pkl, 'wb') as f:
        pkl.dump(dict_all_results, f)