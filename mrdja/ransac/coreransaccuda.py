from numba import cuda, float32
import math
import mrdja.ransac.coreransac as crs
import mrdja.ransac.coreransacutils as crsu
import mrdja.sampling as sampling
import numpy as np
import numba
from time import time
from typing import Tuple, List, Dict
import sys
import mrdja.geometry as geom

@cuda.jit
def get_how_many_and_which_below_threshold_kernel(points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray,
                                         a: float, b: float, c: float, d: float,
                                         optimized_threshold: float, point_indices: np.ndarray) -> None:
    """
    Computes the number of points that are below a threshold distance from a plane using CUDA parallel processing.
    
    :param points_x: The x-coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: The y-coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: The z-coordinates of the points.
    :type points_z: np.ndarray
    :param a: The first coefficient of the plane equation.
    :type a: float
    :param b: The second coefficient of the plane equation.
    :type b: float
    :param c: The third coefficient of the plane equation.
    :type c: float
    :param d: The fourth coefficient of the plane equation.
    :type d: float
    :param optimized_threshold: The threshold distance from the plane.
    :type optimized_threshold: float
    :param point_indices: The array of indices representing the points that are below the threshold.
    :type point_indices: np.ndarray
    """
    i = cuda.grid(1)
    if i < points_x.shape[0]:
        dist = math.fabs(a * points_x[i] + b * points_y[i] + c * points_z[i] + d)
        if dist <= optimized_threshold:
            point_indices[i] = 1

@cuda.jit
def get_how_many_below_threshold_kernel(points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray,
                                         a: float, b: float, c: float, d: float,
                                         optimized_threshold: float, result: int) -> None:
    """
    Computes the number of points that are below a threshold distance from a plane using CUDA parallel processing.
    
    :param points_x: The x-coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: The y-coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: The z-coordinates of the points.
    :type points_z: np.ndarray
    :param a: The first coefficient of the plane equation.
    :type a: float
    :param b: The second coefficient of the plane equation.
    :type b: float
    :param c: The third coefficient of the plane equation.
    :type c: float
    :param d: The fourth coefficient of the plane equation.
    :type d: float
    :param optimized_threshold: The threshold distance from the plane.
    :type optimized_threshold: float
    :param point_indices: The array of indices representing the points that are below the threshold.
    :type point_indices: np.ndarray
    """
    i = cuda.grid(1) # to compute the index of the current thread
    if i < points_x.shape[0]:
        dist = math.fabs(a * points_x[i] + b * points_y[i] + c * points_z[i] + d)
        if dist <= optimized_threshold:
            cuda.atomic.add(result, 0, 1)

@cuda.jit
def get_how_many_line_below_threshold_kernel(points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray,
                                         line_two_points: np.ndarray,
                                         threshold: float, result: int) -> None:
    """
    Computes the number of points that are below a threshold distance from a plane using CUDA parallel processing.
    
    :param points_x: The x-coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: The y-coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: The z-coordinates of the points.
    :type points_z: np.ndarray
    :param a: The first coefficient of the plane equation.
    :type a: float
    :param b: The second coefficient of the plane equation.
    :type b: float
    :param c: The third coefficient of the plane equation.
    :type c: float
    :param d: The fourth coefficient of the plane equation.
    :type d: float
    :param optimized_threshold: The threshold distance from the plane.
    :type optimized_threshold: float
    :param point_indices: The array of indices representing the points that are below the threshold.
    :type point_indices: np.ndarray
    """
    i = cuda.grid(1)
    if i < points_x.shape[0]:
        B = line_two_points[0]
        C = line_two_points[1]
        V_x = points_x[i] - B[0]
        V_y = points_y[i] - B[1]
        V_z = points_z[i] - B[2]
        V = (V_x, V_y, V_z)
        W_x = C[0] - B[0]
        W_y = C[1] - B[1]
        W_z = C[2] - B[2]
        W = (W_x, W_y, W_z)
        cross_product_x = V[1] * W[2] - V[2] * W[1]
        cross_product_y = V[2] * W[0] - V[0] * W[2]
        cross_product_z = V[0] * W[1] - V[1] * W[0]
        
        magnitude_cross_product = math.sqrt(cross_product_x * cross_product_x + cross_product_y * cross_product_y + cross_product_z * cross_product_z)
        magnitude_C_minus_B = math.sqrt((C[0] - B[0]) * (C[0] - B[0]) + (C[1] - B[1]) * (C[1] - B[1]) + (C[2] - B[2]) * (C[2] - B[2]))
        
        dist = magnitude_cross_product / magnitude_C_minus_B
        if dist <= threshold:
            cuda.atomic.add(result, 0, 1)
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

def get_how_many_below_threshold_between_line_and_points_cuda(
    points: np.ndarray, d_points_x: np.ndarray, d_points_y: np.ndarray, d_points_z: np.ndarray,
    line_two_points: Tuple[Tuple[float, float, float], Tuple[float, float, float]], threshold: float) -> Tuple[int, List[int]]:

    """
    Computes the number of points that are below a threshold distance from a plane and their indices using CUDA parallel processing.
    
    :param points: The array of points in the format (x, y, z).
    :type points: np.ndarray
    :param points_x: The x-coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: The y-coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: The z-coordinates of the points.
    :type points_z: np.ndarray
    :param d_points_x: The x-coordinates of the points in device memory.
    :type d_points_x: np.ndarray
    :param d_points_y: The y-coordinates of the points in device memory.
    :type d_points_y: np.ndarray
    :param d_points_z: The z-coordinates of the points in device memory.
    :type d_points_z: np.ndarray
    :param plane: The coefficients of the plane equation.
    :type plane: Tuple[float, float, float, float]
    :param threshold: The threshold distance from the plane.
    :type threshold: float
    :return: The number of points below the threshold and their indices.
    :rtype: Tuple[int, List[int]]
    """
    t1 = time()
    num_points = points.shape[0]
    # Output variable to store the result
    result = np.array([0], dtype=np.int32)
    d_result = cuda.to_device(result)
    threadsperblock = 1024
    max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    threadsperblock = min(max_threads_per_block, threadsperblock)
    blockspergrid = math.ceil(num_points / threadsperblock)
    t2 = time()
    get_how_many_line_below_threshold_kernel[blockspergrid, threadsperblock](d_points_x, d_points_y, d_points_z,
                                         line_two_points, threshold, d_result)
    t3 = time()
    # Copy the result back to the host
    cuda.synchronize()
    result = d_result.copy_to_host()[0]
    return result

# OK
def get_how_many_and_which_below_threshold_between_plane_and_points_and_their_indices_cuda(
    points: np.ndarray, d_points_x: np.ndarray, d_points_y: np.ndarray, d_points_z: np.ndarray, 
    plane: Tuple[float, float, float, float], threshold: float) -> Tuple[int, List[int]]:
    """
    Computes the number of points that are below a threshold distance from a plane and their indices using CUDA parallel processing.
    
    :param points: The array of points in the format (x, y, z).
    :type points: np.ndarray
    :param points_x: The x-coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: The y-coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: The z-coordinates of the points.
    :type points_z: np.ndarray
    :param d_points_x: The x-coordinates of the points in device memory.
    :type d_points_x: np.ndarray
    :param d_points_y: The y-coordinates of the points in device memory.
    :type d_points_y: np.ndarray
    :param d_points_z: The z-coordinates of the points in device memory.
    :type d_points_z: np.ndarray
    :param plane: The coefficients of the plane equation.
    :type plane: Tuple[float, float, float, float]
    :param threshold: The threshold distance from the plane.
    :type threshold: float
    :return: The number of points below the threshold and their indices.
    :rtype: Tuple[int, List[int]]
    """
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
    get_how_many_and_which_below_threshold_kernel[blockspergrid, threadsperblock](d_points_x, d_points_y, d_points_z, a, b, c, d, optimized_threshold, d_point_indices)
    point_indices = d_point_indices.copy_to_host()
    # get the count of point_indices that are not -1
    count = np.count_nonzero(point_indices != -1)
    # get the indices of the points that are not -1
    new_indices = np.where(point_indices != -1)
    new_indices = new_indices[0].tolist()
    return count, new_indices

# OK
def get_how_many_below_threshold_between_plane_and_points_cuda(
    points: np.ndarray, d_points_x: np.ndarray, d_points_y: np.ndarray, d_points_z: np.ndarray, 
    plane: Tuple[float, float, float, float], threshold: float) -> int:
    """
    Computes the number of points that are below a threshold distance from a plane and their indices using CUDA parallel processing.
    
    :param points: The array of points in the format (x, y, z).
    :type points: np.ndarray
    :param points_x: The x-coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: The y-coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: The z-coordinates of the points.
    :type points_z: np.ndarray
    :param d_points_x: The x-coordinates of the points in device memory.
    :type d_points_x: np.ndarray
    :param d_points_y: The y-coordinates of the points in device memory.
    :type d_points_y: np.ndarray
    :param d_points_z: The z-coordinates of the points in device memory.
    :type d_points_z: np.ndarray
    :param plane: The coefficients of the plane equation.
    :type plane: Tuple[float, float, float, float]
    :param threshold: The threshold distance from the plane.
    :type threshold: float
    :return: The number of points below the threshold and their indices.
    :rtype: Tuple[int, List[int]]
    """
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]
    num_points = points.shape[0]
    optimized_threshold = threshold * math.sqrt(a * a + b * b + c * c)
    # Output variable to store the result
    result = np.array([0], dtype=np.int32)
    d_result = cuda.to_device(result)
    threadsperblock = 1024
    max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    threadsperblock = min(max_threads_per_block, threadsperblock)
    blockspergrid = math.ceil(num_points / threadsperblock)
    get_how_many_below_threshold_kernel[blockspergrid, threadsperblock](d_points_x, d_points_y, d_points_z, a, b, c, d, optimized_threshold, d_result)
    # Copy the result back to the host
    cuda.synchronize()
    result = d_result.copy_to_host()[0]
    return result

def get_ransac_line_iteration_results_cuda(points: np.ndarray, 
                                       d_points_x: cuda.devicearray.DeviceNDArray, 
                                       d_points_y: cuda.devicearray.DeviceNDArray, 
                                       d_points_z: cuda.devicearray.DeviceNDArray, 
                                       threshold: float,
                                       random_points: np.ndarray) -> dict:
    """
    Computes the number of inliers and the plane parameters for one iteration of the RANSAC algorithm using CUDA.

    :param points: Array of points.
    :type points: np.ndarray
    :param points_x: X coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: Y coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: Z coordinates of the points.
    :type points_z: np.ndarray
    :param d_points_x: Device array of X coordinates of the points.
    :type d_points_x: cuda.devicearray.DeviceNDArray
    :param d_points_y: Device array of Y coordinates of the points.
    :type d_points_y: cuda.devicearray.DeviceNDArray
    :param d_points_z: Device array of Z coordinates of the points.
    :type d_points_z: cuda.devicearray.DeviceNDArray
    :param num_points: Number of random points to select for each iteration.
    :type num_points: int
    :param threshold: Maximum distance to the plane.
    :type threshold: float
    :return: Dictionary with the plane parameters, the number of inliers, and the indices of the inliers.
    :rtype: dict
    """
    # this takes a lot of time
    if random_points is None:
        current_random_points = sampling.sampling_np_arrays_from_enumerable(points, cardinality_of_np_arrays=2, number_of_np_arrays=1, num_source_elems=len(points), seed=None)[0]
    else:
        current_random_points = random_points
    current_line = (tuple(current_random_points[0]), tuple(current_random_points[1]))
    how_many_in_line = get_how_many_below_threshold_between_line_and_points_cuda(points, d_points_x, d_points_y, d_points_z, current_line, threshold)
    return {"current_line": current_random_points, "threshold": threshold, "number_inliers": how_many_in_line}

def get_ransac_iteration_results_cuda(points: np.ndarray, 
                                       points_x: np.ndarray, 
                                       points_y: np.ndarray, 
                                       points_z: np.ndarray, 
                                       d_points_x: cuda.devicearray.DeviceNDArray, 
                                       d_points_y: cuda.devicearray.DeviceNDArray, 
                                       d_points_z: cuda.devicearray.DeviceNDArray, 
                                       num_points: int, 
                                       threshold: float) -> dict:
    """
    Computes the number of inliers and the plane parameters for one iteration of the RANSAC algorithm using CUDA.

    :param points: Array of points.
    :type points: np.ndarray
    :param points_x: X coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: Y coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: Z coordinates of the points.
    :type points_z: np.ndarray
    :param d_points_x: Device array of X coordinates of the points.
    :type d_points_x: cuda.devicearray.DeviceNDArray
    :param d_points_y: Device array of Y coordinates of the points.
    :type d_points_y: cuda.devicearray.DeviceNDArray
    :param d_points_z: Device array of Z coordinates of the points.
    :type d_points_z: cuda.devicearray.DeviceNDArray
    :param num_points: Number of random points to select for each iteration.
    :type num_points: int
    :param threshold: Maximum distance to the plane.
    :type threshold: float
    :return: Dictionary with the plane parameters, the number of inliers, and the indices of the inliers.
    :rtype: dict
    """
    # esto es lo que tarda mucho
    current_random_points = crs.get_np_array_of_three_random_points_from_np_array_of_points(points, num_points)
    current_plane = geom.get_plane_from_list_of_three_points(current_random_points.tolist())
    how_many_in_plane, current_point_indices = get_how_many_below_threshold_between_plane_and_points_and_their_indices_cuda(points, points_x, points_y, points_z, d_points_x, d_points_y, d_points_z, current_plane, threshold)
    return {"current_plane": current_plane, "number_inliers": how_many_in_plane, "indices_inliers": current_point_indices}


def get_ransac_results_cuda(points, num_points, threshold, num_iterations):
    """
    Computes the best plane that fits a collection of points and the indices of the inliers.
    
    :param points: 3D coordinates of the points as a numpy array with shape (num_points, 3).
    :type points: np.ndarray
    :param num_points: Number of points to use for RANSAC.
    :type num_points: int
    :param threshold: Maximum distance to the plane.
    :type threshold: float
    :param num_iterations: Number of iterations to compute the best plane.
    :type num_iterations: int
    :return: A dictionary with keys "best_plane", "number_inliers", and "indices_inliers".
             "best_plane" is a numpy array with shape (4,) representing the best-fit plane in the form of [a, b, c, d],
             where the equation of the plane is ax + by + cz + d = 0.
             "number_inliers" is an int representing the number of inliers that fit the best-fit plane.
             "indices_inliers" is a numpy array with shape (num_inliers,) representing the indices of the inliers.
    :rtype: dict

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
    best_plane = None
    number_points_in_best_plane = 0
    points_x = np.ascontiguousarray(points[:, 0])
    points_y = np.ascontiguousarray(points[:, 1])
    points_z = np.ascontiguousarray(points[:, 2])
    d_points_x = cuda.to_device(points_x)
    d_points_y = cuda.to_device(points_y)
    d_points_z = cuda.to_device(points_z)
    indices_inliers = None
    for _ in range(num_iterations):
        dict_results = get_ransac_iteration_results_cuda(points, points_x, points_y, points_z, d_points_x, d_points_y, d_points_z, num_points, threshold)
        current_plane = dict_results["current_plane"]
        how_many_in_plane = dict_results["number_inliers"]
        current_indices_inliers = dict_results["indices_inliers"]
        if how_many_in_plane > number_points_in_best_plane:
            # inliers_ratio = how_many_in_plane / num_points
            # max_num_iterations = crsu.compute_number_iterations(inliers_ratio, alpha = 0.05)
            # print("Current inliers ratio: ", inliers_ratio, " Max num iterations: ", max_num_iterations)
            number_points_in_best_plane = how_many_in_plane
            best_plane = current_plane
            indices_inliers = current_indices_inliers
    if indices_inliers is None:
        indices_inliers = np.empty(0, dtype=np.int64)
    return {"best_plane": best_plane, "number_inliers": number_points_in_best_plane, "indices_inliers": indices_inliers}

def get_fitting_data_from_list_planes_cuda(points: np.ndarray, list_planes: List[np.ndarray], threshold: float) -> List[Dict]:
    '''
    Returns the fitting data for each plane in the list of planes.

    :param points: The collection of points to fit the plane to.
    :type points: np.ndarray
    :param list_planes: The list of planes to fit to the points.
    :type list_planes: List[np.ndarray]
    :param threshold: The maximum distance from a point to the plane for it to be considered an inlier.
    :type threshold: float
    :return: A list of dictionaries containing the plane parameters, number of inliers, and their indices.
    :rtype: List[Dict]

    :Example:

    ::

        >>> import mrdja.ransac.coreransac as coreransac
        >>> import open3d as o3d
        >>> import numpy as np
        >>> import random
        >>> import open3d as o3d
        >>> dataset = o3d.data.OfficePointClouds()
        >>> pcds_offices = []
        >>> for pcd_path in dataset.paths:
        >>>     pcds_offices.append(o3d.io.read_point_cloud(pcd_path))
        >>> office_pcd = pcds_offices[0]
        >>> pcd_points = np.asarray(office_pcd.points)
        >>> threshold = 0.1
        >>> num_iterations = 20
        >>> dict_results = coreransac.get_ransac_plane_results(pcd_points, threshold, num_iterations, seed = 42)
        >>> dict_results
        {'best_plane': array([-0.17535096,  0.45186984, -2.44615646,  5.69205427]),
        'number_inliers': 153798,
        'indices_inliers': array([     0,      1,      2, ..., 248476, 248477, 248478])}
        >>> fitting_data = coreransac.get_fitting_data_from_list_planes(pcd_points, [dict_results["best_plane"]], threshold)
        >>> fitting_data
        [{'plane': array([-0.17535096,  0.45186984, -2.44615646,  5.69205427]),
        'number_inliers': 153798,
        'indices_inliers': array([     0,      1,      2, ..., 248476, 248477, 248478])}]
    ''' 
    # Extract x, y, z coordinates from np_points
    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]
    # Convert points_x, points_y, and points_z to contiguous arrays
    d_points_x = np.ascontiguousarray(points_x)
    d_points_y = np.ascontiguousarray(points_y)
    d_points_z = np.ascontiguousarray(points_z)
    d_points_x = cuda.to_device(d_points_x)
    d_points_y = cuda.to_device(d_points_y)
    d_points_z = cuda.to_device(d_points_z)

    list_fitting_data = []
    for plane in list_planes:
        how_many_in_plane, indices_inliers = get_how_many_and_which_below_threshold_between_plane_and_points_and_their_indices_cuda(points, 
                                                                                                        d_points_x=d_points_x,
                                                                                                        d_points_y=d_points_y,
                                                                                                        d_points_z=d_points_z,
                                                                                                        plane = plane, 
                                                                                                        threshold = threshold)
        list_fitting_data.append({"plane": plane, "number_inliers": how_many_in_plane, "indices_inliers": indices_inliers})
    return list_fitting_data

def get_best_fitting_data_from_list_planes_cuda(points: np.ndarray, list_planes: List[np.ndarray], threshold: float) -> Dict:
    '''
    Returns the fitting data for the best plane in the list of planes.

    :param points: The collection of points to fit the plane to.
    :type points: np.ndarray
    :param list_planes: The list of planes to fit to the points.
    :type list_planes: List[np.ndarray]
    :param threshold: The maximum distance from a point to the plane for it to be considered an inlier.
    :type threshold: float
    :return: A dictionary containing the plane parameters, number of inliers, and their indices.
    :rtype: Dict

    :Example:

    ::

        >>> import mrdja.ransac.coreransac as coreransac
        >>> import open3d as o3d
        >>> import numpy as np
        >>> import random
        >>> import open3d as o3d
        >>> dataset = o3d.data.OfficePointClouds()
        >>> pcds_offices = []
        >>> for pcd_path in dataset.paths:
        >>>     pcds_offices.append(o3d.io.read_point_cloud(pcd_path))
        >>> office_pcd = pcds_offices[0]
        >>> pcd_points = np.asarray(office_pcd.points)
        >>> threshold = 0.1
        >>> num_iterations = 20
        >>> dict_results = coreransac.get_ransac_plane_results(pcd_points, threshold, num_iterations, seed = 42)
        >>> dict_results
        {'best_plane': array([-0.17535096,  0.45186984, -2.44615646,  5.69205427]),
        'number_inliers': 153798,
        'indices_inliers': array([     0,      1,      2, ..., 248476, 248477, 248478])}
        >>> fitting_data = coreransac.get_fitting_data_from_list_planes(pcd_points, [dict_results["best_plane"]], threshold)
        >>> fitting_data
        [{'plane': array([-0.17535096,  0.45186984, -2.44615646,  5.69205427]),
        'number_inliers': 153798,
        'indices_inliers': array([     0,      1,      2, ..., 248476, 248477, 248478])}]
        >>> best_fitting_data = coreransac.get_best_fitting_data_from_list_planes(pcd_points, [dict_results["best_plane"]], threshold)
        >>> best_fitting_data
        {'plane': array([-0.17535096,  0.45186984, -2.44615646,  5.69205427]),
        'number_inliers': 153798,
        'indices_inliers': array([     0,      1,      2, ..., 248476, 248477, 248478])}
    '''
    fitting_data = get_fitting_data_from_list_planes_cuda(points, list_planes, threshold)
    best_fitting_data = max(fitting_data, key=lambda fitting_data: fitting_data["number_inliers"])
    return best_fitting_data