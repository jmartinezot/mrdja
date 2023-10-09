# https://github.com/isl-org/Open3D/issues/5647

import numpy as np
import open3d as o3d

full_check = True

print(f'open3d version: {o3d.__version__}')

# arr1 = np.array([1,2,3,4,8,9])
# arr2 = np.array([3,4,5,6,7])
# print(np.intersect1d(arr1,arr2))
# print(np.intersect1d(arr1,arr2).shape[0]*100.0/arr1.shape[0])

sample_pcd_data = o3d.data.PCDPointCloud()
pcd = o3d.t.io.read_point_cloud(sample_pcd_data.path)
pcd = pcd.voxel_down_sample(voxel_size=0.01)
# print(f'pcd has {pcd.point.positions.shape[0]} points')

iterations = 10
inlier_array = []
for i in range(iterations):
    # print(f"Running ransac iteration {i}")
    # https://github.com/isl-org/Open3D/issues/5647#issuecomment-1305137472
    o3d.utility.random.seed(1)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000,
                                             probability=1)
    inlier_array.append(inliers)
    # print(f'Plane model has {inliers.shape[0]} inliers')
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud = inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

if full_check:
    overlap = [np.intersect1d(np.asarray(inlier_array[0]),np.asarray(x)).shape[0]*100.0/np.asarray(inlier_array[0]).shape[0] for x in inlier_array]
    print(f'Overlapping of inliers {overlap}')
overlap = [np.all(np.asarray(inlier_array[0]) == np.asarray(x)) for x in inlier_array]
print(f'Overlapping of inliers {overlap}')

# for i in range(1,iterations):
#     print(f'Overlapping of inliers {np.intersect1d(np.asarray(inlier_array[0]),np.asarray(inlier_array[i])).shape[0]*100.0/np.asarray(inlier_array[0]).shape[0]}')

# o3d.visualization.draw([inlier_cloud, outlier_cloud])
