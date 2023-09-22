import mrdja.ransac.coreransac as coreransac
import mrdja.ransaclp as ransaclp
import mrdja.pointcloud as pointcloud
import mrdja.geometry as geom
import mrdja.drawing as drawing
import open3d as o3d
import numpy as np
import random

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

ransac_iterations = 200
threshold = 0.02
threshold_pcd = 0.02
seed = 42
RANSAC_iterator = coreransac.get_ransac_line_iteration_results

# filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/conferenceRoom_1/conferenceRoom_1.ply"
# filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/office_1/office_1.ply"
filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/WC_1/WC_1.ply"
RANSAC_data_from_file = ransaclp.get_RANSAC_data_from_file(filename, RANSAC_iterator = RANSAC_iterator, 
                                                           ransac_iterations = ransac_iterations, 
                                                           threshold = threshold, audit_cloud=False, seed = seed)

print(RANSAC_data_from_file)

pcd = o3d.io.read_point_cloud(RANSAC_data_from_file["filename"])
np_points = np.asarray(pcd.points)

ordered_pairs = ransaclp.get_lines_and_number_inliers_ordered_by_number_inliers(RANSAC_data_from_file)

pair_lines_number_inliers = ransaclp.get_lines_and_number_inliers_from_RANSAC_data_from_file(RANSAC_data_from_file)

list_sse_plane = ransaclp.get_list_sse_plane(pair_lines_number_inliers, percentage_best = 0.2)

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

# set random seed
np.random.seed(42)
plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
                                         ransac_n=3,
                                         num_iterations=478)
[a, b, c, d] = plane_model
print("Plane equation: {:.2f}x + {:.2f}y + {:.2f}z + {:.2f} = 0".format(a, b, c, d))
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd, inlier_cloud], window_name="Inliers  " + str(len(inliers)) + " inliers")
print("Number of inliers: " + str(len(inliers)))
print("Number of inliers maximum: " + str(how_many_maximum))



