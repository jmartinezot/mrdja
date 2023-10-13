# python3 one_file_script.py --filename_pcd ~/open3d_data/extract/OfficePointClouds/cloud_bin_0.ply --filename_pkl test2.pkl --threshold 0.02 --repetitions 10 --iterations 100,200,300,400,500,600 --percentage_chosen_lines 0.2 --percentage_chosen_planes 0.05
# python3 /tmp/Github/mrdja/mrdja/experiments/one_file_script.py --filename_pcd /tmp/open3d_data/extract/OfficePointClouds/cloud_bin_0.ply --filename_pkl /tmp/test2.pkl --threshold 0.02 --repetitions 10 --iterations 100,200,300,400,500,600 --percentage_chosen_lines 0.2 --percentage_chosen_planes 0.05
# python3 ~/Github/mrdja/mrdja/experiments/one_file_script.py --filename_pcd ~/open3d_data/extract/OfficePointClouds/cloud_bin_0.ply --threshold 0.02 --iterations 100 --percentage_chosen_lines 0.2 --percentage_chosen_planes 0.05 


import argparse
import mrdja.ransaclpexperiments as experiments
import mrdja.ransaclp as ransaclp
import open3d as o3d
import os
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='RANSAC Experiment Script')
parser.add_argument('--filename_pcd', type=str, help='Input PCD filename')
parser.add_argument('--threshold', type=float, default=0.02, help='RANSAC threshold')
parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
parser.add_argument('--verbosity_level', type=int, default=1, help='Verbosity level')
parser.add_argument('--percentage_chosen_lines', type=float, default=0.2, help='Percentage of chosen lines')
parser.add_argument('--percentage_chosen_planes', type=float, default=0.05, help='Percentage of chosen planes')
parser.add_argument('--cuda', action='store_true', help='Use CUDA')
args = parser.parse_args()

seed = 42

# Process the specified filename_pcd
filename_pcd = args.filename_pcd
threshold = args.threshold
iterations = args.iterations
percentage_chosen_lines = args.percentage_chosen_lines
percentage_chosen_planes = args.percentage_chosen_planes
cuda = args.cuda
verbosity_level = args.verbosity_level

filename_only_file = os.path.basename(filename_pcd)

parameters_experiment = experiments.compute_parameters_ransac_line(iterations, percentage_chosen_lines = percentage_chosen_lines, 
                                                               percentage_chosen_planes = percentage_chosen_planes)

# Load the point cloud
pcd = o3d.io.read_point_cloud(filename_pcd)

# standard RANSAC
o3d.utility.random.seed(42)
ransac_plane, inliers = pcd.segment_plane(distance_threshold=threshold,
                                        ransac_n=3,
                                        num_iterations=iterations)
ransac_number_inliers = len(inliers)
print(f"RANSAC number of inliers: {ransac_number_inliers}")
print(f"RANSAC plane equation: {ransac_plane}")
# show the inliers in red and the outliers in original color
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# RANSAC-LP

# Convert the point cloud to a numpy array
np_points = np.asarray(pcd.points)
np.random.seed(seed)
ransaclp_best_data, ransaclp_full_data = ransaclp.get_ransaclp_data_from_np_points(np_points, ransac_iterations = iterations, 
                                                threshold = threshold,
                                                cuda = cuda,
                                                verbosity_level = verbosity_level, 
                                                inherited_verbose_string = "",
                                                seed = None)
print(ransaclp_best_data)
ransaclp_number_inliers = ransaclp_best_data["number_inliers"]
ransaclp_plane = ransaclp_best_data["plane"]
print(f"RANSAC-LP number of inliers: {ransaclp_number_inliers}")
print(f"RANSAC-LP plane equation: {ransaclp_plane}")
# show the inliers in red and the outliers in original color
inlier_cloud = pcd.select_by_index(ransaclp_best_data["indices_inliers"])
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(ransaclp_best_data["indices_inliers"], invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])





