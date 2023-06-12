import numpy as np
from typing import List, Union, Tuple

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

def get_limits_of_graph_from_limits_of_object(min_x: float, max_x: float, min_y: float, max_y: float) -> Tuple[float]:
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
    :rtype: float
    """
    limit = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y))
    return (-limit, limit, -limit, limit)
