import numpy as np
from typing import List, Union, Tuple
from scipy.linalg import svd
import math

def get_plane_from_list_of_three_points(points: List[List[float]]) -> Union[np.ndarray, None]:
    """
    Get plane in form Ax + By + Cz + D = 0 from list of three points.

    :param points: Points.
    :type points: List[List[float]]
    :return: Plane.
    :rtype: Union[np.ndarray, None]

    :Example:

    ::

        >>> import coreransac
        >>> points = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        >>> plane = customransac.get_plane_from_points(points)
        >>> plane
        array([0, 0, 1, 0])
    """
    p1 = np.array(points[0])
    p2 = np.array(points[1])
    p3 = np.array(points[2])
    v1 = p2 - p1
    v2 = p3 - p1
    if np.allclose(v1, np.zeros_like(v1)) or np.allclose(v2, np.zeros_like(v2)) or np.allclose(v1, v2):
        print("Points are collinear, cannot get plane.")
        return None
    normal = np.cross(v1, v2)
    d = -np.dot(normal, p1)
    A, B, C = normal
    return np.array([A, B, C, d])

def get_limits_of_graph_from_limits_of_object(min_x: float, max_x: float, min_y: float, max_y: float) -> Tuple[float, float, float, float]:
    """
    Get limits of graph from limits of object. The (0,0) point should be in the center of the graph and the object should be whole visible. 
    The visible zone should be square. This is useful for plotting.

    :param min_x: Minimum x.
    :type min_x: float
    :param max_x: Maximum x.
    :type max_x: float
    :param min_y: Minimum y.
    :type min_y: float
    :param max_y: Maximum y.
    :type max_y: float
    :return: Limits of graph.
    :rtype: Tuple[float, float, float, float]
    """
    limit = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y))
    return (-limit, limit, -limit, limit)

def get_limits_of_3d_graph_from_limits_of_object(min_x: float, max_x: float, min_y: float, max_y: float, min_z: float, max_z: float) -> \
                                                Tuple[float, float, float, float, float, float]:
    """
    Get limits of graph from limits of object. The (0,0,0) point should be in the center of the graph and the object should be whole visible. 
    The visible zone should be square. This is useful for plotting.

    :param min_x: Minimum x.
    :type min_x: float
    :param max_x: Maximum x.
    :type max_x: float
    :param min_y: Minimum y.
    :type min_y: float
    :param max_y: Maximum y.
    :type max_y: float
    :param min_z: Minimum z.
    :type min_z: float
    :return: Limits of graph.
    :rtype: Tuple[float, float, float, float, float, float]
    """
    limit = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y), abs(min_z), abs(max_z))
    return (-limit, limit, -limit, limit, -limit, limit)

def get_parallelogram_2d_vertices(center: List[float], normal1: List[float], normal2: List[float], length1: float, length2: float):
    '''
    Get vertices of parallelogram, given center, normal vectors, and lengths.

    :param center: Center.
    :type center: List[float]
    :param normal1: Normal vector 1.
    :type normal1: List[float]
    :param normal2: Normal vector 2.
    :type normal2: List[float]
    :param length1: Length 1.
    :type length1: float
    :param length2: Length 2.
    :type length2: float
    :return: Vertices.
    :rtype: List[List[float]]

    :Example:

    ::
    '''
    center = np.array(center)
    normal1 = np.array(normal1)
    normal2 = np.array(normal2)
    vertex1 = center + normal1 * length1 / 2 + normal2 * length2 / 2
    vertex2 = center - normal1 * length1 / 2 + normal2 * length2 / 2
    vertex3 = center - normal1 * length1 / 2 - normal2 * length2 / 2
    vertex4 = center + normal1 * length1 / 2 - normal2 * length2 / 2
    return [vertex1, vertex2, vertex3, vertex4]

def get_parallelogram_3d_vertices(center: List[float], normal1: List[float], normal2: List[float], normal3: List[float], 
                                  length1: float, length2: float, length3: float)-> List[List[float]]:
    '''
    Get vertices of parallelogram, given center, normal vectors, and lengths.

    :param center: Center.
    :type center: List[float]
    :param normal1: Normal vector 1.
    :type normal1: List[float]
    :param normal2: Normal vector 2.
    :type normal2: List[float]
    :param length1: Length 1.
    :type length1: float
    :param length2: Length 2.
    :type length2: float
    :param length3: Length 3.
    :type length3: float
    :return: Vertices.
    :rtype: List[List[float]]

    :Example:

    ::
    '''
    center = np.array(center)
    normal1 = np.array(normal1)
    normal2 = np.array(normal2)
    normal3 = np.array(normal3)
    vertex1 = center + normal1 * length1 / 2 + normal2 * length2 / 2 + normal3 * length3 / 2
    vertex2 = center - normal1 * length1 / 2 + normal2 * length2 / 2 + normal3 * length3 / 2
    vertex3 = center - normal1 * length1 / 2 - normal2 * length2 / 2 + normal3 * length3 / 2
    vertex4 = center + normal1 * length1 / 2 - normal2 * length2 / 2 + normal3 * length3 / 2

    vertex5 = center + normal1 * length1 / 2 + normal2 * length2 / 2 - normal3 * length3 / 2
    vertex6 = center - normal1 * length1 / 2 + normal2 * length2 / 2 - normal3 * length3 / 2
    vertex7 = center - normal1 * length1 / 2 - normal2 * length2 / 2 - normal3 * length3 / 2
    vertex8 = center + normal1 * length1 / 2 - normal2 * length2 / 2 - normal3 * length3 / 2
    return [vertex1, vertex2, vertex3, vertex4, vertex5, vertex6, vertex7, vertex8]

def get_plane_equation(normal1, normal2, point):
    normal1 = np.array(normal1)
    normal2 = np.array(normal2)
    point = np.array(point)
    # Normalize the vectors
    normal1 = normal1 / np.linalg.norm(normal1)
    normal2 = normal2 / np.linalg.norm(normal2)

    # Calculate the normal vector of the plane
    normal = np.cross(normal1, normal2)
    normal = normal / np.linalg.norm(normal)

    # Calculate the value of D
    D = -np.dot(normal, point)

    # Multiply all coefficients by the reciprocal of D
    A, B, C = normal

    # Return the plane equation coefficients
    return A, B, C, D

def find_closest_plane(points: List[List[float]]) -> Tuple[float]:
    """
    Find the closest plane to a set of points based on Euclidean distance.

    :param points: Points.
    :type points: List[List[float]]
    :return: Plane equation coefficients.
    :rtype: Tuple[float]
    """
    # Center the points around the origin
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Compute the singular value decomposition
    _, _, vh = svd(centered_points)

    # Extract the last row of V^T to get the normal vector of the plane
    plane_normal = vh[-1]

    # Normalize the normal vector
    plane_normal /= np.linalg.norm(plane_normal)

    # Choose a point on the plane as the centroid of the centered points
    plane_point = centroid

    # Compute the D coefficient of the plane equation
    D = -np.dot(plane_normal, plane_point)

    # Return the plane equation coefficients
    return (*plane_normal, D)

# compute the distance from a point to a plane
def get_distance_from_point_to_plane(point:Tuple[float, float, float], plane: Tuple[float, float, float, float])->float:
    '''
    Get the distance from a point to a plane.

    :param point: Point.
    :type point: Tuple[float, float, float]
    :param plane: Plane equation coefficients.
    :type plane: Tuple[float, float, float, float]
    :return: Distance from point to plane.
    :rtype: float

    :Example:

    ::
    '''
    A, B, C, D = plane
    x, y, z = point
    return abs(A * x + B * y + C * z + D) / math.sqrt(A * A + B * B + C * C)

# compute the distance from a np array of points to a plane
def get_distance_from_points_to_plane(points:np.ndarray, plane: Tuple[float, float, float, float])->np.ndarray:
    '''
    Get the distance from a np array of points to a plane.

    :param points: Points.
    :type points: np.ndarray
    :param plane: Plane equation coefficients.
    :type plane: Tuple[float, float, float, float]
    :return: Distance from points to plane.
    :rtype: np.ndarray

    :Example:

    ::
    '''
    A, B, C, D = plane
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return abs(A * x + B * y + C * z + D) / math.sqrt(A * A + B * B + C * C)

def fit_plane_svd(points):
    centroid = np.mean(points, axis=0)
    shifted_points = points - centroid
    _, _, V = np.linalg.svd(shifted_points)
    normal = V[2]
    d = -np.dot(normal, centroid)
    # Compute sum of squared errors
    errors = np.abs(np.dot(points, normal) + d)
    sse = np.sum(errors ** 2)

    return normal[0], normal[1], normal[2], d, sse

def get_angle_between_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def get_angle_between_lines(l1, l2):
    v1 = l1[1] - l1[0]
    v2 = l2[1] - l2[0]
    print("Line 1: ", l1)
    print("Line 2: ", l2)
    print("Vector 1: ", v1)
    print("Vector 2: ", v2)
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))