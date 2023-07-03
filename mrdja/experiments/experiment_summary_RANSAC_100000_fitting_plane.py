import mrdja.ransac.coreransac as coreransac
import mrdja.pointcloud as pointcloud
import mrdja.geometry as geometry
import mrdja.drawing as drawing
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

# database_path = "/home/scpmaotj/Stanford3dDataset_v1.2/"
# get a list with all the ply files under database_path
import glob
import os

database_path = "/home/scpmaotj/Stanford3dDataset_v1.2/"
ply_files = glob.glob(database_path + "/**/*.ply", recursive=True)

print(ply_files)

threshold = 0.02
# seed = 42
# filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/WC_1/WC_1.ply"

for filename in ply_files:

    print("filename", filename)

    dict_all_results = {}
    dict_all_results["filename"] = filename

    pcd = o3d.io.read_point_cloud(filename)
    np_points = np.asarray(pcd.points)

    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                            ransac_n=3,
                                            num_iterations=100000)
    [a, b, c, d] = plane_model
    print("Plane equation: {:.2f}x + {:.2f}y + {:.2f}z + {:.2f} = 0".format(a, b, c, d))
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([pcd, inlier_cloud], window_name="Inliers  " + str(len(inliers)) + " inliers")
    print("Number of inliers: " + str(len(inliers)))

    dict_baseline_results = {"n_points_pcd": len(np_points), "n_inliers_maximum": str(len(inliers)), "best_plane": plane_model, "threshold": threshold}
    dict_all_results["standard_RANSAC_100000"] = dict_baseline_results

    # save the results as a pickle file in the same folder as the filename file; to do so, just change the extension of the file to pkl
    filename_pkl = filename.replace(".ply", "_100000.pkl")
    with open(filename_pkl, 'wb') as f:
        pkl.dump(dict_all_results, f)


  