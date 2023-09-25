'''
**RANSAC** core functions for finding geometric primitives (**planes** and **lines**) in point clouds.

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

If we want to extract the best fitting plane from a pointcloud, we have to call the function :func:`get_ransac_plane_results`, which, given
a pointcloud, the maximum distance from a point to the plane for it to be considered an inlier, and the number of iterations to run the RANSAC algorithm,
returns the best plane parameters, the number of inliers, and their indices. It calls the function :func:`get_ransac_plane_iteration_results` to get the results
of each iteration of the RANSAC algorithm. 

'''

import numpy as np
import open3d as o3d
import math
import mrdja.ransac.coreransacutils as crsu
import mrdja.sampling as sampling
from typing import Optional, List, Tuple, Dict, Union
import mrdja.geometry as geom

def get_how_many_below_threshold_between_line_and_points_and_their_indices(points: np.ndarray, line_two_points: np.ndarray, threshold: np.float32) -> Tuple[int, np.ndarray]:
    """
    Computes how many points are below a threshold distance from a line and returns their count and their indices.

    :param points: The collection of points to measure the distance to the line.
    :type points: np.ndarray
    :param line_two_points: Two points defining the line.
    :type line_two_points: np.ndarray
    :param threshold: Maximum distance to the line.
    :type threshold: np.float32
    :return: Number of points below the threshold distance as their indices.
    :rtype: Tuple[int, np.ndarray]

    :Example:

    ::

        >>> import mrdja.ransac.coreransac as coreransac
        >>> import mrdja.geometry as geom
        >>> import mrdja.drawing as drawing
        >>> import mrdja.matplot3d as plot3d
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> line = np.array([[0, 0, 0], [1, 1, 1]])
        >>> points = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3], [2, 2, 4]])
        >>> threshold = 1
        >>> count, indices_below = coreransac.get_how_many_below_threshold_between_line_and_points_and_their_indices(points, line, threshold)
        >>> count
        5
        >>> indices_below
        array([0, 1, 2, 3, 4])
        >>> points[indices_below]
        array([[-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1],
        [ 2,  2,  2],
        [ 2,  2,  3]])

        >>> # Calculate the minimum and maximum points of the cube.
        >>> cube_min = np.min(points, axis=0)
        >>> cube_max = np.max(points, axis=0)

        >>> # Get the intersection of the line with the cube
        >>> intersection_points = geom.get_intersection_points_of_line_with_cube(line, cube_min, cube_max)

        >>> # Draw the cube
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> drawing.draw_cube(cube_min, cube_max, alpha = 0.1, ax = ax)
        >>> # Draw the segment between the intersection points
        >>> plot3d.draw_segment(intersection_points, color="green", ax = ax)
        >>> # Draw the intersection points
        >>> plot3d.draw_points(intersection_points, color="black", style="X", ax = ax)


        >>> # Get the indices of the points above the threshold
        >>> all_indices = np.arange(len(points))
        >>> indices_above = np.setdiff1d(all_indices, indices_below)
        >>> # Plot the points below the threshold in red
        >>> plot3d.draw_points(points[indices_below], color="red", style="o", ax = ax)
        >>> # Plot the points above the threshold in blue
        >>> plot3d.draw_points(points[indices_above], color="blue", style="o", ax = ax)

        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Y')
        >>> ax.set_zlabel('Z')
        >>> ax.legend()
        >>> plt.show()

    |coreransac_get_how_many_below_threshold_between_line_and_points_and_their_indices_example|

    .. |coreransac_get_how_many_below_threshold_between_line_and_points_and_their_indices_example| image:: ../../_static/images/coreransac_get_how_many_below_threshold_between_line_and_points_and_their_indices_example.png


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

    :param points: The collection of points to measure the distance to the line.
    :type points: np.ndarray
    :param plane: Four parameters defining the plane in the form Ax + By + Cz + D = 0.
    :type plane: np.ndarray
    :param threshold: Maximum distance to the line.
    :type threshold: np.float32
    :return: Number of points below the threshold distance as their indices.
    :rtype: Tuple[int, np.ndarray]

    :Example:

    ::

        >>> import mrdja.ransac.coreransac as coreransac
        >>> import numpy as np
        >>> plane = np.array([1, 1, 1, 0])
        >>> points = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3], [2, 2, 4]])
        >>> threshold = 1
        >>> count, indices = coreransac.get_how_many_below_threshold_between_plane_and_points_and_their_indices(points, plane, threshold)
        >>> count
        1
        >>> indices
        array([1])
        >>> points[indices]
        array([[0, 0, 0]])
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
    current_random_points = sampling.sampling_np_arrays_from_enumerable(points, cardinality_of_np_arrays=2, number_of_np_arrays=1, num_source_elems=len_points, seed=seed)[0]
    current_line = current_random_points # the easiest way to get the line parameters
    how_many_in_line, current_point_indices = get_how_many_below_threshold_between_line_and_points_and_their_indices(points, current_line, threshold)
    # print(len_points, current_random_points, current_plane, how_many_in_plane)
    return {"current_random_points": current_random_points, "current_line": current_line, "threshold": threshold, "number_inliers": how_many_in_line, "indices_inliers": current_point_indices}


def get_ransac_plane_iteration_results(points: np.ndarray, threshold: float, len_points:Optional[int]=None, seed: Optional[int]=None) -> dict:
    """
    Returns the results of one iteration of the RANSAC algorithm for plane fitting.

    This functions expects a **collection of points**, the **number of points** in the collection, the **maximum distance** from a point
    to the plane for it to be considered an inlier, and the **seed** to initialize the random number generator.
    It returns a dictionary containing the **current plane parameters**, **number of inliers**, and **their indices**. The keys of the dictionary
    are **current_random_points**, **"current_plane"**, **"threshold"**, **"number_inliers"**, and **"indices_inliers"**, respectively. 
    The type of the values of the dictionary are **np.ndarray**, **np.ndarray**, **float**, **int**, and **np.ndarray**, respectively.
    
    :param points: The collection of points to fit the plane to.
    :type points: np.ndarray
    :param threshold: The maximum distance from a point to the plane for it to be considered an inlier.
    :type threshold: float
    :param len_points: The number of points in the collection of points.
    :type len_points: Optional[int]
    :param seed: The seed to initialize the random number generator.
    :type seed: Optional[int]
    :return: A dictionary containing the current plane parameters, number of inliers, and their indices, as well as the three random points sampled to create the plane.
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
        >>> dict_results = coreransac.get_ransac_plane_iteration_results(pcd_points, threshold, seed = 42)
        >>> dict_results
        >>> {'current_random_points': array([[1.61072648, 1.83984375, 1.91796875],
        >>> [3.00390625, 2.68674755, 2.01953125],
        >>> [2.10068583, 2.34765625, 2.14453125]]),
        >>> 'current_plane': array([ 0.14030194, -0.2658808 ,  0.29252566, -0.297864  ]),
        >>> 'threshold': 0.1,
        >>> 'number_inliers': 34283,
        >>> 'indices_inliers': array([ 98356, 101924, 101956, ..., 271055, 271245, 271246])}
        >>> inliers = dict_results["indices_inliers"]
        >>> inlier_cloud = office_pcd.select_by_index(inliers)
        >>> inlier_cloud.paint_uniform_color([1.0, 0, 0])
        >>> outlier_cloud = office_pcd.select_by_index(inliers, invert=True)
        >>> o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    |coreransac_get_ransac_plane_iteration_results_example|

    .. |coreransac_get_ransac_plane_iteration_results_example| image:: ../../_static/images/coreransac_get_ransac_plane_iteration_results_example.png

    """
    if len_points is None:
        len_points = len(points)
    if seed is not None:
        np.random.seed(seed)
    current_random_points = sampling.sampling_np_arrays_from_enumerable(points, cardinality_of_np_arrays=3, number_of_np_arrays=1, num_source_elems=len_points, seed=seed)[0]
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
        {'best_plane': array([-0.17535096,  0.45186984, -2.44615646,  5.69205427]),
        'number_inliers': 153798,
        'indices_inliers': array([     0,      1,      2, ..., 248476, 248477, 248478])}
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
        dict_results = get_ransac_plane_iteration_results(points, threshold, len_points)
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


def get_fitting_data_from_list_planes(points: np.ndarray, list_planes: List[np.ndarray], threshold: float) -> List[Dict]:
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
    list_fitting_data = []
    for plane in list_planes:
        how_many_in_plane, indices_inliers = get_how_many_below_threshold_between_plane_and_points_and_their_indices(points, plane, threshold)
        list_fitting_data.append({"plane": plane, "number_inliers": how_many_in_plane, "indices_inliers": indices_inliers})
    return list_fitting_data

def get_best_fitting_data_from_list_planes(points: np.ndarray, list_planes: List[np.ndarray], threshold: float) -> Dict:
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
    fitting_data = get_fitting_data_from_list_planes(points, list_planes, threshold)
    best_fitting_data = max(fitting_data, key=lambda fitting_data: fitting_data["number_inliers"])
    return best_fitting_data