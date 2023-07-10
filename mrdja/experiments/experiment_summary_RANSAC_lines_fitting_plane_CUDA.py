import mrdja.ransac.coreransaccuda as coreransaccuda
import mrdja.ransac.coreransac as coreransac
import mrdja.pointcloud as pointcloud
import mrdja.geometry as geometry
import mrdja.drawing as drawing
import open3d as o3d
import numpy as np
import random
import pickle as pkl
from numba import cuda
import time

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

def get_RANSAC_data_from_file(filename, ransac_iterations, threshold):
    dict_full_results = {}
    dict_full_results["filename"] = filename

    pcd = o3d.io.read_point_cloud(filename)
    number_pcd_points = len(pcd.points)
    dict_full_results["number_pcd_points"] = number_pcd_points
    np_points = np.asarray(pcd.points)

    # Extract x, y, z coordinates from np_points
    points_x = np_points[:, 0]
    points_y = np_points[:, 1]
    points_z = np_points[:, 2]

    # Convert points_x, points_y, and points_z to contiguous arrays
    d_points_x = np.ascontiguousarray(points_x)
    d_points_y = np.ascontiguousarray(points_y)
    d_points_z = np.ascontiguousarray(points_z)

    d_points_x = cuda.to_device(d_points_x)
    d_points_y = cuda.to_device(d_points_y)
    d_points_z = cuda.to_device(d_points_z)

    dict_full_results["ransac_iterations_results"] = []

    # np.random.seed(seed)
    max_number_inliers = 0
    best_iteration_results = None
    random_numbers_pairs = coreransac.get_np_array_of_two_random_points_from_np_array_of_points(np_points, repetitions=ransac_iterations, num_points=number_pcd_points)

    for i in range(ransac_iterations):
        print("Iteration of line fitting", i, "out of", ransac_iterations)
        random_points = random_numbers_pairs[i]
        dict_iteration_results = coreransaccuda.get_ransac_line_iteration_results_cuda(np_points, d_points_x, d_points_y, d_points_z, number_pcd_points, threshold, random_points=random_points)
        if dict_iteration_results["number_inliers"] > max_number_inliers:
            max_number_inliers = dict_iteration_results["number_inliers"]
            best_iteration_results = dict_iteration_results
        dict_full_results["ransac_iterations_results"].append(dict_iteration_results)
    dict_full_results["ransac_best_iteration_results"] = best_iteration_results
    return dict_full_results

def compute_parameters_RANSAC_line (line_iterations):
    number_chosen_lines = int(line_iterations / 5)
    number_lines_pairs = int(number_chosen_lines * (number_chosen_lines - 1) / 2)
    number_chosen_planes = int(number_lines_pairs * 0.05)
    total_iterations = line_iterations + number_chosen_planes
    return number_chosen_lines, number_lines_pairs, number_chosen_planes, total_iterations

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


# get a list with all the ply files under database_path
import glob
import os

database_path = "/home/scpmaotj/Stanford3dDataset_v1.2/"
# database_path = "/home/bee/S3DIS/Stanford3dDataset_v1.2/"
ply_files = glob.glob(database_path + "/**/*.ply", recursive=True)

print(ply_files)


threshold = 0.02
# seed = 42
# filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/WC_1/WC_1.ply"

# filename_test = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/WC_1/WC_1.ply"

# ply_files = [filename_test]

t1 = time.time()

for filename in ply_files:

    filename_pkl_check = filename.replace(".ply", ".pkl")
    # check if filename_pkl_check exists
    if os.path.isfile(filename_pkl_check):
        continue

    print("filename", filename)

    dict_all_results = {}
    dict_all_results["filename"] = filename

    pcd = o3d.io.read_point_cloud(filename)
    np_points = np.asarray(pcd.points)

    # Extract x, y, z coordinates from np_points
    points_x = np_points[:, 0]
    points_y = np_points[:, 1]
    points_z = np_points[:, 2]

    # Convert points_x, points_y, and points_z to contiguous arrays
    d_points_x = np.ascontiguousarray(points_x)
    d_points_y = np.ascontiguousarray(points_y)
    d_points_z = np.ascontiguousarray(points_z)

    d_points_x = cuda.to_device(d_points_x)
    d_points_y = cuda.to_device(d_points_y)
    d_points_z = cuda.to_device(d_points_z)

    for num_iterations in [100, 200, 300, 400, 500, 600]:
    # for num_iterations in [10, 11, 12]:
        number_chosen_lines, number_lines_pairs, number_chosen_planes, total_iterations = compute_parameters_RANSAC_line(num_iterations)

        # filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/conferenceRoom_1/conferenceRoom_1.ply"
        # filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/office_1/office_1.ply"

        dict_standard_RANSAC_results_list = list()
        dict_line_RANSAC_results_list = list()

        for n_loop in range(10):

            RANSAC_data_from_file = get_RANSAC_data_from_file(filename, num_iterations, threshold)

            print(RANSAC_data_from_file)

            # create a function that takes the RANSAC_data_from_file and the iteration number and returns the inliers
            # create a function that takes the RANSAC_data_from_file and the iteration number and returns the number of inliers

            ordered_pairs = get_lines_and_number_inliers_ordered_by_number_inliers(RANSAC_data_from_file)

            list_sse_plane = []
            current_iteration = 0
            for i in range(number_chosen_lines):
                for j in range(i+1, number_chosen_lines):
                    # compute the current number of iterations
                    current_iteration += 1
                    print("Iteration pairs of lines", current_iteration, "out of", number_lines_pairs, "Loop", n_loop, "out of", 10)
                    line_1 = ordered_pairs[i][0]
                    line_2 = ordered_pairs[j][0]
                    points = np.array([line_1[0], line_1[1], line_2[0], line_2[1]])
                    a, b, c, d, sse = geometry.fit_plane_svd(points)
                    new_plane = np.array([a, b, c, d])
                    list_sse_plane.append((sse, new_plane))

            # get the plane values for the 5% percentile of sse (the best planes)
            list_sse = [pair_sse_plane[0] for pair_sse_plane in list_sse_plane]
            list_plane = [pair_sse_plane[1] for pair_sse_plane in list_sse_plane]
            percentile_05 = np.percentile(list_sse, 5)
            list_sse_05 = [list_sse[i] for i in range(len(list_sse)) if list_sse[i] <= percentile_05]
            list_plane_05 = [list_plane[i] for i in range(len(list_plane)) if list_sse[i] <= percentile_05]
            print (list_sse_05)
            print (list_plane_05)

            # get the best plane from the 95% percentile of sse and the number of inliers

            how_many_maximum = 0
            best_plane = None
            print("Starting the final loop")
            for i in range(len(list_sse_05)):
                print("Iteration best fitting planes", i, "out of", len(list_sse_05), "Loop", n_loop, "out of", 10)
                new_plane = list_plane_05[i]
                how_many = coreransaccuda.get_how_many_below_threshold_between_plane_and_points_cuda(np_points, d_points_x, d_points_y, d_points_z, new_plane, threshold)
                if how_many > how_many_maximum:
                    how_many_maximum = how_many
                    best_plane = new_plane
                    sse_best_plane = list_sse_05[i]
            print (how_many_maximum)
            print (best_plane)
            print (sse_best_plane)
            # o3d.visualization.draw_geometries([pcd, inlier_cloud], window_name="Inliers  " + str(how_many) + " inliers")

            # set random seed
            # np.random.seed(42)
            plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                                    ransac_n=3,
                                                    num_iterations=total_iterations)

            dict_standard_RANSAC_results = {"n_points_pcd": len(np_points), "n_inliers_maximum": len(inliers), "best_plane": plane_model, "threshold": threshold}
            dict_line_RANSAC_results = {"n_points_pcd": len(np_points), "n_inliers_maximum": how_many_maximum, "best_plane": best_plane, "threshold": threshold}
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

    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                            ransac_n=3,
                                            num_iterations=100000)

    dict_baseline_results = {"n_points_pcd": len(np_points), "n_inliers_maximum": len(inliers), "best_plane": plane_model, "threshold": threshold}
    dict_all_results["standard_RANSAC_100000"] = dict_baseline_results

    # save the results as a pickle file in the same folder as the filename file; to do so, just change the extension of the file to pkl
    filename_pkl = filename.replace(".ply", ".pkl")
    with open(filename_pkl, 'wb') as f:
        pkl.dump(dict_all_results, f)

t2 = time.time()
print("Time elapsed:", t2-t1)
  