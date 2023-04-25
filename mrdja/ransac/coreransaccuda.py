from numba import cuda
import math
import coreransac as crs
import coreransacutils as crsu
import numpy as np
import numba
from time import time

@cuda.jit
def get_how_many_below_threshold_kernel(points_x, points_y, points_z, a, b, c, d, optimized_threshold, point_indices):
    i = cuda.grid(1)
    if i < points_x.shape[0]:
        dist = math.fabs(a * points_x[i] + b * points_y[i] + c * points_z[i] + d)
        if dist <= optimized_threshold:
            point_indices[i] = 1

'''
@cuda.jit
def get_how_many_below_threshold_kernel(points_x, points_y, points_z, a, b, c, d, optimized_threshold, point_indices):
    i = cuda.grid(1)
    
    # max_threads_per_block = cuda.config.MAX_THREADS_PER_BLOCK
    # THREADS_PER_BLOCK = min(max_threads_per_block, points_x.shape[0])
    THREADS_PER_BLOCK = 512
    # Define shared memory array
    shared_points_x = cuda.shared.array(shape=(THREADS_PER_BLOCK,), dtype=numba.float32)
    shared_points_y = cuda.shared.array(shape=(THREADS_PER_BLOCK,), dtype=numba.float32)
    shared_points_z = cuda.shared.array(shape=(THREADS_PER_BLOCK,), dtype=numba.float32)
    
    if i < points_x.shape[0]:
        # Copy data from global memory to shared memory
        shared_points_x[cuda.threadIdx.x] = points_x[i]
        shared_points_y[cuda.threadIdx.x] = points_y[i]
        shared_points_z[cuda.threadIdx.x] = points_z[i]
        
        cuda.syncthreads()
        
        # Access data from shared memory
        dist = math.fabs(a * shared_points_x[cuda.threadIdx.x] + b * shared_points_y[cuda.threadIdx.x] + c * shared_points_z[cuda.threadIdx.x] + d)
        
        cuda.syncthreads()
        
        if dist <= optimized_threshold:
            point_indices[i] = 1
'''

def get_how_many_below_threshold_between_plane_and_points_and_their_indices_cuda(points, points_x, points_y, points_z, d_points_x, d_points_y, d_points_z, plane, threshold):
    t1 = time()
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]
    num_points = points.shape[0]
    point_indices = np.zeros(num_points, dtype=np.int32)
    optimized_threshold = threshold * math.sqrt(a * a + b * b + c * c)
    point_indices = np.empty(num_points, dtype=np.int64)
    # fill point_indices with -1
    point_indices[:] = -1
    threadsperblock = 512
    blockspergrid = math.ceil(num_points / threadsperblock)
    # point_indices = cuda.device_array(point_indices.shape, dtype=point_indices.dtype)
    d_point_indices = cuda.to_device(point_indices)
    t2 = time()
    get_how_many_below_threshold_kernel[blockspergrid, threadsperblock](d_points_x, d_points_y, d_points_z, a, b, c, d, optimized_threshold, d_point_indices)
    t3 = time()
    point_indices = d_point_indices.copy_to_host()
    # get the count of point_indices that are not -1
    count = np.count_nonzero(point_indices != -1)
    # get the indices of the points that are not -1
    new_indices = np.where(point_indices != -1)
    new_indices = new_indices[0].tolist()
    t4 = time()
    print(f't2 - t1: {(t2-t1):.4f}s')
    print(f't3 - t2: {(t3-t2):.4f}s')
    print(f't4 - t3: {(t4-t3):.4f}s')
    return count, new_indices

def get_ransac_iteration_results_cuda(points, points_x, points_y, points_z, d_points_x, d_points_y, d_points_z, num_points, threshold):
    t1 = time()
    # esto es lo que tarda mucho
    current_random_points = crs.get_np_array_of_three_random_points_from_np_array_of_points(points, num_points)
    t2 = time()
    current_plane = crs.get_plane_from_list_of_three_points(current_random_points.tolist())
    t3 = time()
    print(f't2 - t1: {(t2-t1):.4f}s')
    print(f't3 - t2: {(t3-t2):.4f}s')
    how_many_in_plane, current_point_indices = get_how_many_below_threshold_between_plane_and_points_and_their_indices_cuda(points, points_x, points_y, points_z, d_points_x, d_points_y, d_points_z, current_plane, threshold)
    print(num_points, current_random_points, current_plane, how_many_in_plane)
    return {"current_plane": current_plane, "number_inliers": how_many_in_plane, "indices_inliers": current_point_indices}


def get_ransac_results_cuda(points, num_points, threshold, num_iterations):
    """
    Computes the best plane that fits a collection of points and the indices of the inliers.
    
    :param points_x: X coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: Y coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: Z coordinates of the points.
    :type points_z: np.ndarray
    :param threshold: Maximum distance to the plane.
    :type threshold: np.float32
    :param num_iterations: Number of iterations to compute the best plane.
    :type num_iterations: np.int64
    :param random_points_indices_1: Indices of the points to use as first random point in each iteration.
    :type random_points_indices_1: np.ndarray
    :param random_points_indices_2: Indices of the points to use as second random point in each iteration.
    :type random_points_indices_2: np.ndarray
    :param random_points_indices_3: Indices of the points to use as third random point in each iteration.
    :type random_points_indices_3: np.ndarray
    :return: Best plane parameters and the indices of the points that fit the plane.
    :rtype: Tuple[np.ndarray, np.ndarray]

    :Example:

    ::

        >>> import customransac
        >>> import open3d as o3d
        >>> import numpy as np
        >>> import random
        >>> pcd_filename = "/tmp/Lantegi/Kubic.ply"
        >>> pcd = o3d.io.read_point_cloud(pcd_filename)
        >>> pcd_points = np.asarray(pcd.points, dtype="float32")
        >>> num_points = len(pcd_points)
        >>> pcd_points_x = pcd_points[:, 0]
        >>> pcd_points_y = pcd_points[:, 1]
        >>> pcd_points_z = pcd_points[:, 2]
        >>> threshold = 0.1
        >>> num_iterations = 2
        >>> list_num_points = list(range(num_points))
        >>> random_points_indices_1 = np.array([], dtype="int64")
        >>> random_points_indices_2 = np.array([], dtype="int64")
        >>> random_points_indices_3 = np.array([], dtype="int64")
        >>> for i in range(num_iterations):
        >>>     random_points_indices = random.sample(list_num_points, 3)
        >>>     random_points_indices_1 = np.append(random_points_indices_1, random_points_indices[0])
        >>>     random_points_indices_2 = np.append(random_points_indices_2, random_points_indices[1])
        >>>     random_points_indices_3 = np.append(random_points_indices_3, random_points_indices[2])
        >>> plane_parameters, indices = customransac.get_best_plane_and_inliers(pcd_points_x, pcd_points_y, pcd_points_z, threshold, num_iterations, random_points_indices_1, random_points_indices_2, random_points_indices_3)
        >>> plane_parameters
        array([ 0.0012,  0.0012,  0.0012,  0.0012])
        >>> inlier_cloud = pcd.select_by_index(inliers)
        >>> inlier_cloud.paint_uniform_color([1.0, 0, 0])
        >>> outlier_cloud = pcd.select_by_index(inliers, invert=True)
        >>> o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    """
    t1 = time()
    best_plane = None
    number_points_in_best_plane = 0
    points_x = np.ascontiguousarray(points[:, 0])
    points_y = np.ascontiguousarray(points[:, 1])
    points_z = np.ascontiguousarray(points[:, 2])
    d_points_x = cuda.to_device(points_x)
    d_points_y = cuda.to_device(points_y)
    d_points_z = cuda.to_device(points_z)
    indices_inliers = None
    t110 = time()
    print("Time to copy data to GPU ", t110 - t1)
    for _ in range(num_iterations):
        t3 = time()
        dict_results = get_ransac_iteration_results_cuda(points, points_x, points_y, points_z, d_points_x, d_points_y, d_points_z, num_points, threshold)
        t5 = time()
        print("Time to call RANSAC iteration CUDA: ", t5 - t3)
        current_plane = dict_results["current_plane"]
        how_many_in_plane = dict_results["number_inliers"]
        current_indices_inliers = dict_results["indices_inliers"]
        t6 = time()
        print("Time to compute extract values from dictionary ", t6 - t5)
        if how_many_in_plane > number_points_in_best_plane:
            # inliers_ratio = how_many_in_plane / num_points
            # max_num_iterations = crsu.compute_number_iterations(inliers_ratio, alpha = 0.05)
            # print("Current inliers ratio: ", inliers_ratio, " Max num iterations: ", max_num_iterations)
            number_points_in_best_plane = how_many_in_plane
            best_plane = current_plane
            indices_inliers = current_indices_inliers
        t4 = time()
        print("Time to compute RANSAC iteration: ", t4 - t3)
    if indices_inliers is None:
        indices_inliers = np.empty(0, dtype=np.int64)
    t2 = time()
    print("Time to compute RANSAC: ", t2 - t1)
    return {"best_plane": best_plane, "number_inliers": number_points_in_best_plane, "indices_inliers": indices_inliers}