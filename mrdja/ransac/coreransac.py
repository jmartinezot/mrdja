'''
We only work with **RANSAC** for finding geometric primitives in point clouds. These primitives are **planes** and **lines**.

In this module all the functions related to the **RANSAC** algorithm for **planes** and **lines** are defined.

The **pointcloud** is expected to be a **numpy array** of shape **(N, 3)** where **N** is the number of **points** and **3** is the 
number of **coordinates (x, y, z)**. 
**No** information about **RGB** values or normals are taken into account. Therefore, an **Open3D** point cloud must be **converted**
to a **numpy array** before 
using the functions in this module.

The following code is an example of how to extract the points from an Open3D point cloud and convert them to a numpy array:

::

    import numpy as np
    import open3d as o3d

    filename = "pointcloud.ply"
    pcd = o3d.io.read_point_cloud(filename)
    np_points = np.asarray(pcd.points)

The pointcloud could be in **ply* or **pcd** format, because Open3D can read both formats.

You can always access to the pointclouds provided by Open3D. For example, the following code shows how to access to the LivingRoomPointClouds, 
which are 57 point clouds of from the Redwood RGB-D Dataset.

::

    import open3d as o3d
    dataset = o3d.data.LivingRoomPointClouds()
    pcds_living_rooms = []
    for pcd_path in dataset.paths:
        pcds_living_rooms.append(o3d.io.read_point_cloud(pcd_path))

Other 53 pointclouds from the same dataset are also available in OfficePointClouds:

::

    import open3d as o3d
    dataset = o3d.data.OfficePointClouds()
    pcds_offices = []
    for pcd_path in dataset.paths:
        pcds_offices.append(o3d.io.read_point_cloud(pcd_path))

After executing the previous code, we could define two examples of a living room pointcloud and an office pointcloud, respectively:

::

    living_room_pcd = pcds_living_rooms[0]
    office_pcd = pcds_offices[0]

    
And visualize them:

::

    o3d.visualization.draw_geometries([living_room_pcd])

    
|coreransac_living_room_pcd|

    .. |coreransac_living_room_pcd| image:: ../../_static/images/coreransac_living_room_pcd.png

::

    o3d.visualization.draw_geometries([office_pcd])

|coreransac_office_pcd|

    .. |coreransac_office_pcd| image:: ../../_static/images/coreransac_office_pcd.png


A brief explanation of the relationship between the functions defined here is given below:

If we want to extract the best fitting plane from a pointcloud, we have to call the function :func:`get_ransac_plane_results`

- :func:`get_np_array_of_two_random_points_from_np_array_of_points`: Returns two random points from a list of points.
- :func:`get_np_array_of_three_random_points_from_np_array_of_points`: Returns three random points from a list of points.
- :func:`get_how_many_below_threshold_between_line_and_points_and_their_indices`: Computes how many points are below a threshold distance from a line and returns their count and their indices.
- :func:`get_how_many_below_threshold_between_plane_and_points_and_their_indices`: Computes how many points are below a threshold distance from a plane and returns their count and their indices.
- :func:`get_pointcloud_from_indices`: Get a point cloud from a list of indices.
- :func:`get_ransac_line_iteration_results`: Returns the results of one iteration of the RANSAC algorithm for line fitting.
- :func:`get_ransac_iteration_results`: Returns the results of one iteration of the RANSAC algorithm for plane fitting.
- :func:`get_ransac_results`: Computes the best plane that fits a collection of points and the indices of the inliers.

'''

import numpy as np
import open3d as o3d
import math
import mrdja.ransac.coreransacutils as crsu
from typing import Optional, List, Tuple, Dict, Union
import mrdja.geometry as geom

def get_np_array_of_two_random_points_from_np_array_of_points(points: np.ndarray, repetitions: int=1, num_points: Optional[int] = None, seed: Optional[int] = None) -> np.ndarray:
    """
    Returns two random points from a list of points.

    :param points: Points.
    :type points: np.ndarray
    :param num_points: Number of points.
    :type num_points: Optional[int]
    :return: Three random points.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import coreransac
        >>> import numpy as np
        >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        >>> points
        array([[0, 0, 0],
               [1, 0, 0],
               [0, 1, 0]])
        >>> random_points = coreransac.get_three_random_points(points)
        >>> random_points
        array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 0]])
    """
    if num_points is None:
        num_points = len(points)
    if seed is not None:
        np.random.seed(seed)
    random_points_indices = np.random.choice(range(num_points), size= 2 * repetitions, replace=False)
    random_points = points[random_points_indices]
    # create a list of pairs of points from the array random_points
    random_points = random_points.reshape(repetitions, 2, 3)
    return random_points

def get_np_array_of_three_random_points_from_np_array_of_points(points: np.ndarray, num_points: Optional[int] = None, seed: Optional[int] = None) -> np.ndarray:
    """
    Returns three random points from a list of points.

    :param points: Points.
    :type points: np.ndarray
    :param num_points: Number of points.
    :type num_points: Optional[int]
    :return: Three random points.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import coreransac
        >>> import numpy as np
        >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        >>> points
        array([[0, 0, 0],
               [1, 0, 0],
               [0, 1, 0]])
        >>> random_points = coreransac.get_three_random_points(points)
        >>> random_points
        array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 0]])
    """
    if num_points is None:
        num_points = len(points)
    if seed is not None:
        np.random.seed(seed)
    random_points_indices = np.random.choice(range(num_points), 3, replace=False)
    random_points = points[random_points_indices]
    return random_points

def get_how_many_below_threshold_between_line_and_points_and_their_indices(points: np.ndarray, line_two_points: np.ndarray, threshold: np.float32) -> Tuple[int, np.ndarray]:
    """
    Computes how many points are below a threshold distance from a line and returns their count and their indices.

    :param a: A parameter of the plane.
    :type a: np.float32
    :param b: B parameter of the plane.
    :type b: np.float32
    :param c: C parameter of the plane.
    :type c: np.float32
    :param d: D parameter of the plane.
    :type d: np.float32
    :param points_x: X coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: Y coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: Z coordinates of the points.
    :type points_z: np.ndarray
    :param threshold: Maximum distance to the plane.
    :type threshold: np.float32
    :return: Number of points below the threshold distance as their indices.
    :rtype: Tuple[int, np.ndarray]

    :Example:

    ::

        >>> import customransac
        >>> plane = np.array([1,1,1,1], dtype="float32")
        >>> points = [[-1, -1, -1], [0, 0, 0], [1, 1, 1], [2, 2, 2]]
        >>> points_x = np.array([p[0] for p in points], dtype="float32")
        >>> points_y = np.array([p[1] for p in points], dtype="float32")
        >>> points_z = np.array([p[2] for p in points], dtype="float32")
        >>> threshold = 1
        >>> count, indices = customransac.get_how_many_below_threshold_between_plane_and_points_and_indices(plane[0], plane[1], plane[2], plane[3], points_x, points_y, points_z, threshold)
        >>> count
        1
        >>> indices
        array([1])
    """
    B = line_two_points[0]
    C = line_two_points[1]
    # the distance between a point A and the line defined by the points B and C can be computed as 
    # magnitude(cross(A - B, C - B)) / magnitude(C - B)
    # https://math.stackexchange.com/questions/1905533/find-perpendicular-distance-from-point-to-line-in-3d

    cross_product = np.cross(points - B, C - B)
    magnitude_cross_product = np.linalg.norm(cross_product, axis=1)
    magnitude_C_minus_B = np.linalg.norm(C - B)
    distance = magnitude_cross_product / magnitude_C_minus_B

    indices_inliers = np.array([index for index, value in enumerate(distance) if value <= threshold], dtype=np.int64)
    return len(indices_inliers), indices_inliers

def get_how_many_below_threshold_between_plane_and_points_and_their_indices(points: np.ndarray, plane: np.ndarray, threshold: np.float32) -> Tuple[int, np.ndarray]:
    """
    Computes how many points are below a threshold distance from a plane and returns their count and their indices.

    :param a: A parameter of the plane.
    :type a: np.float32
    :param b: B parameter of the plane.
    :type b: np.float32
    :param c: C parameter of the plane.
    :type c: np.float32
    :param d: D parameter of the plane.
    :type d: np.float32
    :param points_x: X coordinates of the points.
    :type points_x: np.ndarray
    :param points_y: Y coordinates of the points.
    :type points_y: np.ndarray
    :param points_z: Z coordinates of the points.
    :type points_z: np.ndarray
    :param threshold: Maximum distance to the plane.
    :type threshold: np.float32
    :return: Number of points below the threshold distance as their indices.
    :rtype: Tuple[int, np.ndarray]

    :Example:

    ::

        >>> import customransac
        >>> plane = np.array([1,1,1,1], dtype="float32")
        >>> points = [[-1, -1, -1], [0, 0, 0], [1, 1, 1], [2, 2, 2]]
        >>> points_x = np.array([p[0] for p in points], dtype="float32")
        >>> points_y = np.array([p[1] for p in points], dtype="float32")
        >>> points_z = np.array([p[2] for p in points], dtype="float32")
        >>> threshold = 1
        >>> count, indices = customransac.get_how_many_below_threshold_between_plane_and_points_and_indices(plane[0], plane[1], plane[2], plane[3], points_x, points_y, points_z, threshold)
        >>> count
        1
        >>> indices
        array([1])
    """
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]
    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]
    # the distance between the point P with coordinates (xo, yo, zo) and the given plane with equation Ax + By + Cz = D, 
    # is given by |Axo + Byo+ Czo + D|/√(A^2 + B^2 + C^2).
    denominator = math.sqrt(a * a + b * b + c * c)
    optimized_threshold = threshold * denominator  # for not computing the division for each point
    distance = np.abs(points_x * a + points_y * b + points_z * c + d)
    # point_indices = np.where(distance <= optimized_threshold)[0]
    indices_inliers = np.array([index for index, value in enumerate(distance) if value <= optimized_threshold], dtype=np.int64)
    return len(indices_inliers), indices_inliers

def get_pointcloud_from_indices(pcd: o3d.geometry.PointCloud, indices: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Get a point cloud from a list of indices.
    
    :param pcd: Point cloud.
    :type pcd: open3d.geometry.PointCloud
    :param indices: Indices.
    :type indices: np.ndarray
    :return: Point cloud.
    :rtype: open3d.geometry.PointCloud

    :Example:

    ::

        >>> import open3d as o3d
        >>> import customransac
        >>> pcd = o3d.io.read_point_cloud("tests/data/fragment.ply")
        >>> indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> pcd = customransac.get_pointcloud_from_indices(pcd, indices)
        >>> len(pcd.points)
        10
    """
    pcd = pcd.select_by_index(indices)
    return pcd

def get_ransac_line_iteration_results(points: np.ndarray, threshold: float, len_points:Optional[int]=None, seed: Optional[int]=None) -> dict:
    """
    Returns the results of one iteration of the RANSAC algorithm for line fitting.
    
    :param points: The collection of points to fit the line to.
    :type points: np.ndarray
    :param threshold: The maximum distance from a point to the line for it to be considered an inlier.
    :type threshold: float
    :param len_points: The number of points in the collection of points.
    :type len_points: Optional[int]
    :return: A dictionary containing the current line parameters, number of inliers, and their indices.
    :rtype: dict
    """
    if len_points is None:
        len_points = len(points)
    if seed is not None:
        np.random.seed(seed)
    current_random_points = get_np_array_of_two_random_points_from_np_array_of_points(points, len_points)
    current_line = current_random_points # the easiest way to get the line parameters
    how_many_in_line, current_point_indices = get_how_many_below_threshold_between_line_and_points_and_their_indices(points, current_line, threshold)
    # print(len_points, current_random_points, current_plane, how_many_in_plane)
    return {"current_random_points": current_random_points, "current_line": current_line, "threshold": threshold, "number_inliers": how_many_in_line, "indices_inliers": current_point_indices}


def get_ransac_iteration_results(points: np.ndarray, threshold: float, len_points:Optional[int]=None, seed: Optional[int]=None) -> dict:
    """
    Returns the results of one iteration of the RANSAC algorithm for plane fitting.
    
    :param points: The collection of points to fit the plane to.
    :type points: np.ndarray
    :param threshold: The maximum distance from a point to the plane for it to be considered an inlier.
    :type threshold: float
    :param len_points: The number of points in the collection of points.
    :type len_points: Optional[int]
    :param seed: The seed to initialize the random number generator.
    :type seed: Optional[int]
    :return: A dictionary containing the current plane parameters, number of inliers, and their indices.
    :rtype: dict
    """
    if len_points is None:
        len_points = len(points)
    if seed is not None:
        np.random.seed(seed)
    current_random_points = get_np_array_of_three_random_points_from_np_array_of_points(points, len_points, seed)
    current_plane = geom.get_plane_from_list_of_three_points(current_random_points.tolist())
    how_many_in_plane, current_point_indices = get_how_many_below_threshold_between_plane_and_points_and_their_indices(points, current_plane, threshold)
    # print(len_points, current_random_points, current_plane, how_many_in_plane)
    return {"current_random_points": current_random_points, "current_plane": current_plane, "threshold": threshold, "number_inliers": how_many_in_plane, "indices_inliers": current_point_indices}
    

def get_ransac_plane_results(points: np.ndarray, threshold: float, num_iterations: int, len_points:Optional[int]=None, seed: Optional[int] = None) -> dict:
    """
    Computes the **best plane** that fits a **collection of points** and the indices of the inliers.

    This functions expects a **collection of points**, the **number of points** in the collection, the **maximum distance** from a point 
    to the plane for it to be considered an inlier, and the **number of iterations** to run the RANSAC algorithm. 
    It returns a **dictionary** containing the
    **best plane parameters**, **number of inliers**, and **their indices**. The keys of the dictionary are **"best_plane"**, 
    **"number_inliers"**, and
    **"indices_inliers"**, respectively. The type of the values of the dictionary are **np.ndarray**, **int**, and **np.ndarray**, respectively. 
    Reproducibility is enforced by setting the **seed** parameter to a fixed value.

    :param points: The collection of points to fit the plane to.
    :type points: np.ndarray
    :param threshold: The maximum distance from a point to the plane for it to be considered an inlier.
    :type threshold: float
    :param num_iterations: The number of iterations to run the RANSAC algorithm.
    :type num_iterations: int
    :param len_points: The number of points in the collection of points.
    :type len_points: Optional[int]
    :param seed: The seed to initialize the random number generator.
    :type seed: Optional[int]
    :return: A dictionary containing the best plane parameters, number of inliers, and their indices.
    :rtype: dict

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
        >>> {'best_plane': array([-0.17535096,  0.45186984, -2.44615646,  5.69205427]),
        >>> 'number_inliers': 153798,
        >>> 'indices_inliers': array([     0,      1,      2, ..., 248476, 248477, 248478])}
        >>> inliers = dict_results["indices_inliers"]
        >>> inlier_cloud = office_pcd.select_by_index(inliers)
        >>> inlier_cloud.paint_uniform_color([1.0, 0, 0])
        >>> outlier_cloud = office_pcd.select_by_index(inliers, invert=True)
        >>> o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    |coreransac_get_ransac_plane_results_example|

    .. |coreransac_get_ransac_plane_results_example| image:: ../../_static/images/coreransac_get_ransac_plane_results_example.png


    """
    if len_points is None:
        len_points = len(points)
    if seed is not None:
        np.random.seed(seed)
    best_plane = None
    number_points_in_best_plane = 0
    for _ in range(num_iterations):
        dict_results = get_ransac_iteration_results(points, threshold, len_points)
        current_plane = dict_results["current_plane"]
        how_many_in_plane = dict_results["number_inliers"]
        current_indices_inliers = dict_results["indices_inliers"]
        if how_many_in_plane > number_points_in_best_plane:
            inliers_ratio = how_many_in_plane / len_points
            max_num_iterations = crsu.compute_number_iterations(inliers_ratio, alpha = 0.05)
            print("Current inliers ratio: ", inliers_ratio, " Max num iterations: ", max_num_iterations)
            number_points_in_best_plane = how_many_in_plane
            best_plane = current_plane
            indices_inliers = current_indices_inliers.copy()
    return {"best_plane": best_plane, "number_inliers": number_points_in_best_plane, "indices_inliers": indices_inliers}


