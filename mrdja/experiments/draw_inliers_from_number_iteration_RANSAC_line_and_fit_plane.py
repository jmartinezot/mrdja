import mrdja.ransac.coreransac as coreransac
import mrdja.pointcloud as pointcloud
import mrdja.geometry as geometry
import mrdja.drawing as drawing
import open3d as o3d
import numpy as np

'''
Line-guided plane detection in pointclouds.
Tomar, por ejemplo, 200 rectas y luego los 40 mejores por parejas. Eso son 780 parejas. Tomar el 5% mejor según el error de ajuste del plano.
Eso son 39 parejas. En total 239 veces que hay que mirar los inliers mirando todos los puntos.
Comparar con las iteraciones de RANSAC que se necesitan para obtener el mismo o mayor número de inliers, un 5% menos, un 10% menos, etc.
También poner la diferencia entre los dos al llegar ambos al mismo número de iteraciones.
Comenzar con la misma semilla aleatoria para que las iteraciones de RANSAC sean las mismas.
'''

def partitions_with_respect_to_vector(reference_vector, list_vectors, angle_threshold):
    # returns a list of lists of vectors
    # list_vectors is traversed only once
    list_partitions = []
    for vector in list_vectors:
        # if the angle between the vector and all the vectors in some element of list_partitions is less than angle_threshold, 
        # then add the vector to that element; otherwise, create a new element with the reference_vector and vector

        # compute the angle between the vector and all the vectors in list_partitions
        for partition in list_partitions:
            print(partition)
            print(vector)
            all_angles = [geometry.get_angle_between_vector(vector, partition_vector) for partition_vector in partition]
            if all(abs(angle) < angle_threshold for angle in all_angles):
                partition.append(vector)
                break
        else:
            list_partitions.append([reference_vector, vector])
    return list_partitions

def partitions_with_respect_to_line(reference_line, list_lines, angle_threshold):
    # returns a list of lists of vectors
    # list_vectors is traversed only once
    list_partitions = []
    for index, line in enumerate(list_lines):
        # if the angle between the vector and all the vectors in some element of list_partitions is less than angle_threshold, 
        # then add the vector to that element; otherwise, create a new element with the reference_vector and vector

        # compute the angle between the vector and all the vectors in list_partitions
        for partition in list_partitions:
            print(partition)
            print(line)
            all_angles = [geometry.get_angle_between_lines(line, partition_line[0]) for partition_line in partition]
            if all(abs(angle) < angle_threshold for angle in all_angles):
                partition.append([line, index+1])
                break
        else:
            list_partitions.append([[reference_line, 0], [line, index+1]])
    return list_partitions

# create an example
reference_vector = np.array([1, 0, 0])
list_vectors = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 1, 1]), np.array([1, 1, 1])]
angle_threshold = 0.1
# list_partitions = partitions_with_respect_to_vector(reference_vector, list_vectors, angle_threshold)

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
    for _ in range(ransac_iterations):
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

def get_inliers_from_iteration(RANSAC_data_from_file, np_points, iteration_number):
    dict_iteration_results = RANSAC_data_from_file["ransac_iterations_results"][iteration_number]
    inliers = np_points[dict_iteration_results["indices_inliers"]]
    pcd_inliers = o3d.geometry.PointCloud()
    pcd_inliers.points = o3d.utility.Vector3dVector(inliers)
    pcd_inliers.paint_uniform_color([1, 0, 0])
    return pcd_inliers

def get_number_inliers_from_iteration(RANSAC_data_from_file, iteration_number):
    dict_iteration_results = RANSAC_data_from_file["ransac_iterations_results"][iteration_number]
    return dict_iteration_results["number_inliers"]

# using open3d create a geometry with the inliers painted red, and show the inliers, the original point cloud and the number of iteration and the number
# of inliers in the same window; create a for loop that shows the inliers for each iteration

def draw_inliers_from_iteration(RANSAC_data_from_file, pcd, iteration_number):
    pcd_inliers = get_inliers_from_iteration(RANSAC_data_from_file, np_points, iteration_number)
    number_inliers = get_number_inliers_from_iteration(RANSAC_data_from_file, iteration_number)
    o3d.visualization.draw_geometries([pcd, pcd_inliers], window_name="Inliers from iteration " + str(iteration_number) + " with " + str(number_inliers) + " inliers")

for iteration_number in range(ransac_iterations):
    # only show if the number of inliers is greater than threshold
    threshold_number_inliers = 1000
    if get_number_inliers_from_iteration(RANSAC_data_from_file, iteration_number) > threshold_number_inliers:
        draw_inliers_from_iteration(RANSAC_data_from_file, pcd, iteration_number)

# create a function that takes the RANSAC_data_from_file and returns the number of inliers for each iteration

def get_number_inliers_from_RANSAC_data_from_file(RANSAC_data_from_file):
    number_inliers = []
    for dict_iteration_results in RANSAC_data_from_file["ransac_iterations_results"]:
        number_inliers.append(dict_iteration_results["number_inliers"])
    return number_inliers

number_inliers = get_number_inliers_from_RANSAC_data_from_file(RANSAC_data_from_file)

# create an histogram of the number of inliers for each iteration with matplotlib

import matplotlib.pyplot as plt

plt.hist(number_inliers, bins=100)
plt.show()

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

# get the first 50 pairs and get all the pairs of pairs

how_many_maximum = 0
list_sse_inliers = []
for i in range(40):
    for j in range(i+1, 40):
        line_1 = ordered_pairs[i][0]
        line_2 = ordered_pairs[j][0]
        points = np.array([line_1[0], line_1[1], line_2[0], line_2[1]])
        a, b, c, d, sse = geometry.fit_plane_svd(points)
        new_plane = np.array([a, b, c, d])
        threshold_pcd = 0.02
        how_many, inliers = coreransac.get_how_many_below_threshold_between_plane_and_points_and_their_indices(np_points, new_plane, threshold_pcd)
        list_sse_inliers.append((sse, how_many))
        if how_many > how_many_maximum:
            how_many_maximum = how_many
        if how_many > 200000:
            inliers = np.asarray(inliers)
            inlier_cloud = pcd.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([pcd, inlier_cloud], window_name="Inliers from iteration " + str(i) + " and " + str(j) + " with " + str(how_many) + " inliers")
print (how_many_maximum)

#plot the relationship between sse and number of inliers

list_sse_inliers = sorted(list_sse_inliers, key=lambda pair_sse_inliers: pair_sse_inliers[0])
list_sse = [pair_sse_inliers[0] for pair_sse_inliers in list_sse_inliers]
list_inliers = [pair_sse_inliers[1] for pair_sse_inliers in list_sse_inliers]
plt.plot(list_sse, list_inliers)
plt.show()

# plot a histogram of the sse

list_sse = [pair_sse_inliers[0] for pair_sse_inliers in list_sse_inliers]
plt.hist(list_sse, bins=100)
plt.show()

# get the elements of list_sse below the percentil 0.05 of list_sse

percentil = 0.05
list_sse = [pair_sse_inliers[0] for pair_sse_inliers in list_sse_inliers]
list_sse = sorted(list_sse)
print (list_sse[: int(len(list_sse)*percentil)])



# count how_many sse are below 0.05, 0.1, 0.2, 0.3, 0.4, 0.5

threshold_sse = 0.2
how_many_below_threshold = 0
for sse in list_sse:
    if sse < threshold_sse:
        how_many_below_threshold += 1
print (how_many_below_threshold)

# plot a histogram of the number of inliers

list_inliers = [pair_sse_inliers[1] for pair_sse_inliers in list_sse_inliers]
plt.hist(list_inliers, bins=100)
plt.show()

# plot a scatter plot of the sse and the number of inliers restricted to the sse values between 0 and 0.1

list_sse = [pair_sse_inliers[0] for pair_sse_inliers in list_sse_inliers]
list_inliers = [pair_sse_inliers[1] for pair_sse_inliers in list_sse_inliers]
plt.scatter(list_sse, list_inliers)
plt.xlim(0, 0.1)
plt.ylim(0, 100000)
plt.show()

# plot a scatter plot of the sse and the number of inliers restricted to the sse values below the 0.05 percentile

list_sse = [pair_sse_inliers[0] for pair_sse_inliers in list_sse_inliers]
list_inliers = [pair_sse_inliers[1] for pair_sse_inliers in list_sse_inliers]
percentile_05 = np.percentile(list_sse, 5)
list_sse_05 = [list_sse[i] for i in range(len(list_sse)) if list_sse[i] < percentile_05]
list_inliers_05 = [list_inliers[i] for i in range(len(list_sse)) if list_sse[i] < percentile_05]
plt.scatter(list_sse_05, list_inliers_05)
plt.xlim(0, 0.0002)
plt.ylim(0, 300000)
plt.show()

# plot a boxplot of the sse and the number of inliers restricted to the sse values between 0 and 0.1

list_sse = [pair_sse_inliers[0] for pair_sse_inliers in list_sse_inliers]
list_inliers = [pair_sse_inliers[1] for pair_sse_inliers in list_sse_inliers]
plt.boxplot(list_sse)
plt.ylim(-1, 10)
plt.show()

# get the sse values for the 95% percentile of the number of inliers

list_sse = [pair_sse_inliers[0] for pair_sse_inliers in list_sse_inliers]
list_inliers = [pair_sse_inliers[1] for pair_sse_inliers in list_sse_inliers]
percentile_95 = np.percentile(list_inliers, 95)
list_sse_95 = [list_sse[i] for i in range(len(list_sse)) if list_inliers[i] > percentile_95]
list_inliers_95 = [list_inliers[i] for i in range(len(list_inliers)) if list_inliers[i] > percentile_95]
print (list_sse_95)
print (list_inliers_95)

'''
angle_threshold = 0.3
how_many_maximum = 0
ordered_lines = [ordered_pair[0] for ordered_pair in ordered_pairs]
# ordered_vectors = [ordered_line[1] - ordered_line[0] for ordered_line in ordered_lines]
while len(ordered_lines) > 1:
    reference_line = ordered_lines[0]
    partitions = partitions_with_respect_to_line(reference_line, ordered_lines[1:], angle_threshold)
    print("Partitions: " + str(len(partitions)))
    print (partitions)
    # get the partition with the most elements
    partition_with_most_elements = max(partitions, key=lambda partition: len(partition))
    # get the lines and the indices from partition_with_most_elements
    lines = [line for line, index in partition_with_most_elements]
    indices = [index for line, index in partition_with_most_elements]
    # get the points of the partition with the most elements
    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])
    points = np.asarray(points)
    # remove the partition with the most elements from the list of ordered_lines
    # sort the indices in descending order
    indices = sorted(indices, reverse=True)
    for index in indices:
        ordered_lines.pop(index)

    # get the plane that fits the points
    new_plane = geometry.fit_plane_svd(points)
    # get the number of inliers
    threshold_pcd = 0.02
    how_many, inliers = coreransac.get_how_many_below_threshold_between_plane_and_points_and_their_indices(np_points, new_plane, threshold_pcd)
    if how_many > how_many_maximum:
        how_many_maximum = how_many
    inliers = np.asarray(inliers)
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd, inlier_cloud], window_name="Inliers with " + str(how_many) + " inliers")
print (how_many_maximum)
'''
