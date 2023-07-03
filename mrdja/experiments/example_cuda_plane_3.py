import numpy as np
import open3d as o3d
from time import time
import math
from typing import Tuple, List
from numba import cuda
import mrdja.ransac.coreransaccuda as coreransaccuda


@cuda.jit
def get_how_many_below_threshold_kernel(points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray,
                                         a: float, b: float, c: float, d: float,
                                         optimized_threshold: float, result: int) -> None:
    i = cuda.grid(1) # to compute the index of the current thread
    if i < points_x.shape[0]:
        dist = math.fabs(a * points_x[i] + b * points_y[i] + c * points_z[i] + d)
        if dist <= optimized_threshold:
            cuda.atomic.add(result, 0, 1)

filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/WC_1/WC_1.ply"
pcd = o3d.io.read_point_cloud(filename)
np_points = np.asarray(pcd.points)
threshold_pcd = 22.52

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

num_points = np_points.shape[0]
iterations = 1000

threadsperblock = 1024
max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK
threadsperblock = min(max_threads_per_block, threadsperblock)
blockspergrid = math.ceil(num_points / threadsperblock)

t1 = time()
np.random.seed(42)
for _ in range(iterations):
    a, b, c, d = np.random.rand(4)
    # a, b, c, d = 0.1, 0.2, 0.3, 0.4
    new_plane = (a, b, c, d)

    # Output variable to store the result
    result = np.array([0], dtype=np.int32)
    d_result = cuda.to_device(result)

    optimized_threshold = threshold_pcd * math.sqrt(a * a + b * b + c * c)
    get_how_many_below_threshold_kernel[blockspergrid, threadsperblock](d_points_x, d_points_y, d_points_z, a, b, c, d, optimized_threshold, d_result)

    # Copy the result back to the host
    cuda.synchronize()
    result = d_result.copy_to_host()[0]
    # print(result)

t2 =time()
print(f't2 - t1: {(t2-t1):.4f}s')

t1 = time()
np.random.seed(42)
for _ in range(iterations):
    a, b, c, d = np.random.rand(4)
    # a, b, c, d = 0.1, 0.2, 0.3, 0.4
    new_plane = (a, b, c, d)
    # Output variable to store the result
    result = np.array([0], dtype=np.int32)
    d_result = cuda.to_device(result)
    other_result = coreransaccuda.get_how_many_below_threshold_between_plane_and_points_and_their_indices_cuda(np_points, d_points_x, d_points_y, d_points_z, new_plane, threshold_pcd)
    # print(other_result)
t2 =time()
print(f't2 - t1: {(t2-t1):.4f}s')

t1 = time()
plane_model, inliers = pcd.segment_plane(distance_threshold=threshold_pcd,
                                         ransac_n=3,
                                         num_iterations=1000)
t2 =time()
print(f't2 - t1: {(t2-t1):.4f}s')

t1 = time()
plane_model, inliers = pcd.segment_plane(distance_threshold=threshold_pcd,
                                         ransac_n=3,
                                         num_iterations=100000)
t2 =time()
print(f't2 - t1: {(t2-t1):.4f}s')
