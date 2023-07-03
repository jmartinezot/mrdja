import numpy as np
import open3d as o3d
from time import time
import math
from typing import Tuple, List
from numba import cuda

@cuda.jit
def get_how_many_below_threshold_kernel(points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray,
                                         a: float, b: float, c: float, d: float, a1: float, b1: float, c1: float, d1: float,
                                         optimized_threshold: float, point_indices: np.ndarray, point_indices_1: np.ndarray) -> None:
    i = cuda.grid(1)
    if i < points_x.shape[0]:
        dist = math.fabs(a * points_x[i] + b * points_y[i] + c * points_z[i] + d)
        # if dist <= optimized_threshold:
        #     point_indices[i] = 1
        point_indices[i] = int(dist <= optimized_threshold) * 2 - 1
        dist1 = math.fabs(a1 * points_x[i] + b1 * points_y[i] + c1 * points_z[i] + d1)
        # if dist1 <= optimized_threshold:
        #    point_indices_1[i] = 1
        point_indices_1[i] = int(dist1 <= optimized_threshold) * 2 - 1

def get_how_many_below_threshold_between_plane_and_points_and_their_indices_cuda(
    num_points: int, d_points_x: np.ndarray, d_points_y: np.ndarray, d_points_z: np.ndarray, 
    plane: Tuple[float, float, float, float], plane1: Tuple[float, float, float, float], threshold: float) -> Tuple[int, List[int]]:
    # t1 = time()
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]
    a1 = plane1[0]
    b1 = plane1[1]
    c1 = plane1[2]
    d1 = plane1[3]
    optimized_threshold = threshold * math.sqrt(a * a + b * b + c * c)
    point_indices = np.empty(num_points, dtype=np.int64)
    point_indices_1 = np.empty(num_points, dtype=np.int64)
    # fill point_indices with -1
    point_indices[:] = -1
    threadsperblock = 512
    blockspergrid = math.ceil(num_points / threadsperblock)
    # point_indices = cuda.device_array(point_indices.shape, dtype=point_indices.dtype)
    d_point_indices = cuda.to_device(point_indices)
    d_point_indices_1 = cuda.to_device(point_indices_1)
    # t2 = time()
    get_how_many_below_threshold_kernel[blockspergrid, threadsperblock](d_points_x, d_points_y, d_points_z, a, b, c, d, a1, b1, c1, d1, optimized_threshold, d_point_indices, d_point_indices_1)
    # t3 = time()
    point_indices = d_point_indices.copy_to_host()
    # get the count of point_indices that are not -1
    count = np.count_nonzero(point_indices != -1)
    # get the indices of the points that are not -1
    new_indices = np.where(point_indices != -1)
    new_indices = new_indices[0].tolist()
    '''
    t4 = time()
    print(f't2 - t1: {(t2-t1):.4f}s')
    print(f't3 - t2: {(t3-t2):.4f}s')
    print(f't4 - t3: {(t4-t3):.4f}s')
    print(f't4 - t1: {(t4-t1):.4f}s')
    '''
    return count, new_indices

filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/WC_1/WC_1.ply"

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

num_points = np_points.shape[0]

t1 = time()
for i in range(500):
    a, b, c, d = np.random.rand(4)
    a1, b1, c1, d1 = np.random.rand(4)
    new_plane = (a, b, c, d)
    new_plane1 = (a1, b1, c1, d1)
    threshold_pcd = 0.1

    how_many, inliers = get_how_many_below_threshold_between_plane_and_points_and_their_indices_cuda(num_points, 
                                                                                                    d_points_x,
                                                                                                    d_points_y,
                                                                                                    d_points_z,
                                                                                                    new_plane, new_plane1, threshold_pcd)
t2 =time()
print(f't2 - t1: {(t2-t1):.4f}s')