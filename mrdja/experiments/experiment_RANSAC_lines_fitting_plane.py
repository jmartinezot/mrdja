import mrdja.ransac.coreransac as coreransac
import mrdja.pointcloud as pointcloud
import mrdja.geometry as geometry
import mrdja.drawing as drawing
import open3d as o3d
import numpy as np
import random

'''
Line-guided plane detection in pointclouds.
Tomar, por ejemplo, 200 rectas y luego los 40 mejores por parejas. Eso son 780 parejas. Tomar el 5% mejor según el error de ajuste del plano.
Eso son 39 parejas. En total 239 veces que hay que mirar los inliers mirando todos los puntos.
Comparar con las iteraciones de RANSAC que se necesitan para obtener el mismo o mayor número de inliers, un 5% menos, un 10% menos, etc.
También poner la diferencia entre los dos al llegar ambos al mismo número de iteraciones.
Comenzar con la misma semilla aleatoria para que las iteraciones de RANSAC sean las mismas.
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

def get_RANSAC_data_from_file(filename, ransac_iterations, threshold, seed):
    dict_full_results = {}
    dict_full_results["filename"] = filename

    pcd = o3d.io.read_point_cloud(filename)
    '''
    audit_before_sanitizing = pointcloud.pointcloud_audit(pcd)
    dict_full_results["audit_before_sanitizing"] = audit_before_sanitizing
    pcd = pointcloud.pointcloud_sanitize(pcd)
    audit_after_sanitizing = pointcloud.pointcloud_audit(pcd)
    dict_full_results["audit_after_sanitizing"] = audit_after_sanitizing
    '''
    number_pcd_points = len(pcd.points)
    dict_full_results["number_pcd_points"] = number_pcd_points
    np_points = np.asarray(pcd.points)

    dict_full_results["ransac_iterations_results"] = []

    np.random.seed(seed)
    max_number_inliers = 0
    best_iteration_results = None
    for i in range(ransac_iterations):
        print("Iteration", i)
        dict_iteration_results = coreransac.get_ransac_line_iteration_results(np_points, threshold, number_pcd_points)
        if dict_iteration_results["number_inliers"] > max_number_inliers:
            max_number_inliers = dict_iteration_results["number_inliers"]
            best_iteration_results = dict_iteration_results
        dict_full_results["ransac_iterations_results"].append(dict_iteration_results)
    dict_full_results["ransac_best_iteration_results"] = best_iteration_results
    return dict_full_results

ransac_iterations = 200
threshold = 0.02
seed = 42

filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/conferenceRoom_1/conferenceRoom_1.ply"
RANSAC_data_from_file = get_RANSAC_data_from_file(filename, ransac_iterations, threshold, seed)

print(RANSAC_data_from_file)

# create a function that takes the RANSAC_data_from_file and the iteration number and returns the inliers
# create a function that takes the RANSAC_data_from_file and the iteration number and returns the number of inliers

pcd = o3d.io.read_point_cloud(RANSAC_data_from_file["filename"])
np_points = np.asarray(pcd.points)

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

ordered_pairs = get_lines_and_number_inliers_ordered_by_number_inliers(RANSAC_data_from_file)

list_sse_plane = []
current_iteration = 0
for i in range(40):
    for j in range(i+1, 40):
        # compute the current number of iterations
        current_iteration += 1
        print("Iteration", current_iteration)
        line_1 = ordered_pairs[i][0]
        line_2 = ordered_pairs[j][0]
        points = np.array([line_1[0], line_1[1], line_2[0], line_2[1]])
        a, b, c, d, sse = geometry.fit_plane_svd(points)
        new_plane = np.array([a, b, c, d])
        threshold_pcd = 0.02
        list_sse_plane.append((sse, new_plane))

# get the plane values for the 5% percentile of sse (the best planes)
list_sse = [pair_sse_plane[0] for pair_sse_plane in list_sse_plane]
list_plane = [pair_sse_plane[1] for pair_sse_plane in list_sse_plane]
percentile_05 = np.percentile(list_sse, 10)
list_sse_05 = [list_sse[i] for i in range(len(list_sse)) if list_sse[i] <= percentile_05]
list_plane_05 = [list_plane[i] for i in range(len(list_plane)) if list_sse[i] <= percentile_05]
print (list_sse_05)
print (list_plane_05)

# get the best plane from the 95% percentile of sse and the number of inliers

how_many_maximum = 0
best_plane = None
print("Starting the final loop")
for i in range(len(list_sse_05)):
    print("Iteration", i)
    new_plane = list_plane_05[i]
    threshold_pcd = 0.02
    how_many, inliers = coreransac.get_how_many_below_threshold_between_plane_and_points_and_their_indices(np_points, new_plane, threshold_pcd)
    if how_many > how_many_maximum:
        how_many_maximum = how_many
        inliers_maximum = inliers
        best_plane = new_plane
        sse_best_plane = list_sse_05[i]
print (how_many_maximum)
print (best_plane)
print (sse_best_plane)
inliers = np.asarray(inliers_maximum)
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd, inlier_cloud], window_name="Inliers  " + str(how_many) + " inliers")

list_values = [how_many_maximum, how_many_maximum*0.95, how_many_maximum*0.9]

def get_iteration_number_when_a_computed_value_is_reached(list_values, max_iteration_number, pcd_points, threshold, len_points):
    iteration_number_dict = {}
    # copy the list of values
    list_compared_values = list(list_values)
    iteration_number = 0
    while list_compared_values != [] and iteration_number < max_iteration_number:
        print("Iteration", iteration_number)
        print("list_compared_values", list_compared_values)
        print("iteration_number_dict", iteration_number_dict)
        print("max_iteration_number", max_iteration_number)
        iteration_number += 1
        # get a random value from 0 to 10000
        dict_results = coreransac.get_ransac_iteration_results(pcd_points, threshold, len_points)
        current_plane = dict_results["current_plane"]
        number_inliers = dict_results["number_inliers"]
        print("current_plane", current_plane)
        print("number_inliers", number_inliers)
        index = 0
        while index < len(list_compared_values):
            value = list_compared_values[index]
            if number_inliers >= value:
                list_compared_values.pop(index)
                iteration_number_dict[value] = iteration_number
            else:
                index += 1
    return iteration_number_dict

get_iteration_number_when_a_computed_value_is_reached(list_values, 1000, np_points, 0.02, len(np_points))

