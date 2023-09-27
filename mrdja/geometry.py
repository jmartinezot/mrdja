import numpy as np
from typing import List, Union, Tuple, Optional
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

def get_parallelepiped_3d_vertices(center: List[float], normal1: List[float], normal2: List[float], normal3: List[float], 
                                  length1: float, length2: float, length3: float)-> List[List[float]]:
    '''
    Get vertices of parallelepiped, given center, normal vectors, and lengths.

    :param center: Center.
    :type center: List[float]
    :param normal1: Normal vector 1.
    :type normal1: List[float]
    :param normal2: Normal vector 2.
    :type normal2: List[float]
    :param normal3: Normal vector 3.
    :type normal3: List[float]
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

def get_parallelogram_3d_vertices(center: List[float], normal1: List[float], normal2: List[float],  
                                  length1: float, length2: float)-> List[List[float]]:
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

def get_intersection_points_of_line_with_cube(line: np.ndarray, cube_min: np.ndarray, cube_max: np.ndarray) -> np.ndarray:
    '''
    Get the intersection points of a line with a cube.

    :param line: Line described as two points.
    :type line: np.ndarray
    :param cube_min: Minimum point of the cube.
    :type cube_min: np.ndarray
    :param cube_max: Maximum point of the cube.
    :type cube_max: np.ndarray
    :return: Intersection points.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import mrdja.geometry as geom
        >>> import mrdja.drawing as drawing
        >>> import numpy as np
        >>> line = np.array([[0, 0, 0], [1, 1, 1]])
        >>> cube_min = np.array([-2, -2, -1])
        >>> cube_max = np.array([1, 2, 2])
        >>> intersection_points = geom.get_intersection_points_of_line_with_cube(line, cube_min, cube_max)
        >>> intersection_points
        array([[ 1.,  1.,  1.],
              [-1., -1., -1.]])
    '''
    # Initialize an empty list to store intersection points.
    intersection_points = []

    # Define the planes representing the faces of the cube.
    # Each plane is defined as (A, B, C, D), where Ax + By + Cz + D = 0.
    # We'll calculate the six planes for the cube.
    
    # Plane equations for the cube faces (front, back, top, bottom, left, right).
    planes = [
        (1, 0, 0, -cube_min[0]),  # Front face
        (-1, 0, 0, cube_max[0]),  # Back face
        (0, 1, 0, -cube_min[1]),  # Top face
        (0, -1, 0, cube_max[1]),  # Bottom face
        (0, 0, 1, -cube_min[2]),  # Left face
        (0, 0, -1, cube_max[2])   # Right face
    ]

    # Check for intersection with each plane.
    for plane in planes:
        intersection_point = get_intersection_point_of_line_with_plane(line, plane)
        if intersection_point is not None:
            # Check if the intersection point is not already in the list.
            if not any(np.allclose(intersection_point, point) for point in intersection_points):
                # Check if the intersection point is within the bounds of the cube.
                if all(cube_min <= intersection_point) and all(intersection_point <= cube_max):
                    intersection_points.append(intersection_point)

    return np.array(intersection_points)


def get_intersection_point_of_line_with_plane(line: np.ndarray, plane: np.ndarray) -> Optional[np.ndarray]:
    '''
    Get the intersection point of a line with a plane. If it is parallel, return None.

    :param line: Line described as two points.
    :type line: np.ndarray
    :param plane: Plane described as Ax + By + Cz + D = 0.
    :type plane: np.ndarray
    :return: Intersection point.
    :rtype: Optional[np.ndarray]

    :Example:

    ::

        >>> import mrdja.geometry as geom
        >>> import numpy as np 
        >>> import mrdja.drawing as drawing
        >>> line = np.array([[0, 0, 0], [1, 1, 1]])
        >>> plane = np.array([0, 0, 1, -3])
        >>> intersection_point = geom.get_intersection_point_of_line_with_plane(line, plane)
        >>> intersection_point
        array([3., 3., 3.])
        >>> drawing.draw_line_extension_to_plane(line, plane)

    |drawing_draw_line_extension_to_plane_example|

    .. |drawing_draw_line_extension_to_plane_example| image:: ../../_static/images/drawing_draw_line_extension_to_plane_example.png

    '''
    # Step 1: Calculate the direction vector of the line.
    direction = line[1] - line[0]
    
    # Step 2: Calculate the normal vector of the plane.
    normal = plane[:3]
    
    # Step 3: Check if the line is parallel to the plane.
    dot_product = np.dot(direction, normal)
    if np.isclose(dot_product, 0, atol=1e-6):
        return None
    
    # Step 4: Calculate the parameter t for the line's equation.
    t = - (np.dot(normal, line[0]) + plane[3]) / dot_product
    
    # Step 5: Calculate the intersection point.
    intersection_point = line[0] + t * direction
    return intersection_point

def get_two_perpendicular_unit_vectors_in_plane(plane: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
    '''
    Get two perpendicular unit vectors in a plane.

    :param plane: Plane described as Ax + By + Cz + D = 0.
    :type plane: np.ndarray
    :return: Two perpendicular unit vectors in the plane.
    :rtype: Tuple[np.ndarray, np.ndarray]

    :Example:

    ::

        >>> import mrdja.geometry as geom
        >>> import numpy as np
        >>> plane = np.array([0, 0, 1, -3])
        >>> perpendicular1, perpendicular2 = geom.get_two_perpendicular_unit_vectors_in_plane(plane)
        >>> perpendicular1
        array([0., 1., 0.])
        >>> perpendicular2
        array([-1., 0., 0.])
    '''
    # Step 1: Calculate the normal vector of the plane.
    normal = plane[:3]
    # Step 2: Calculate the first perpendicular vector.
    if normal[0] == 0 and normal[1] == 0:
        perpendicular1 = np.array([0, 1, 0])
    else:
        perpendicular1 = np.array([normal[1], -normal[0], 0])
    # Step 3: Calculate the second perpendicular vector.
    perpendicular2 = np.cross(normal, perpendicular1)
    # Step 4: Normalize the vectors.
    perpendicular1 = perpendicular1 / np.linalg.norm(perpendicular1)
    perpendicular2 = perpendicular2 / np.linalg.norm(perpendicular2)
    return perpendicular1, perpendicular2

def get_best_plane_from_points_from_two_segments(segment_1: np.ndarray, segment_2: np.ndarray) -> Tuple[np.ndarray, float]:
    '''
    Computes the best fitting plane to the four points of two segments.

    :param segment_1: The first segment.
    :type segment_1: np.ndarray
    :param segment_2: The second segment.
    :type segment_2: np.ndarray
    :return: The best fitting plane and the sum of squared errors.
    :rtype: Tuple[np.ndarray, float]

    :Example:

    ::

        >>> import mrdja.geometry as geom
        >>> import numpy as np
        >>> segment_1 = np.array([[0, 0, 0], [1, 0, 0]])
        >>> segment_2 = np.array([[0, 1, 0], [1, 1, 0]])
        >>> geom.get_best_plane_from_points_from_two_segments(segment_1, segment_2)
        (array([ 0.,  0.,  1., -0.]), 0.0)
        >>> # another example
        >>> segment_1 = np.array([[1, 2, 3], [4, 5, 6]])
        >>> segment_2 = np.array([[7, 8, 9], [10, 11, 12]])
        >>> geom.get_best_plane_from_points_from_two_segments(segment_1, segment_2)
        (array([ 0.81649658, -0.40824829, -0.40824829,  1.22474487]),
        1.0107280348144214e-29)
        >>> # another example 
        >>> segment_1 = np.array([[0, 0, 0], [1, 0, 0]])
        >>> segment_2 = np.array([[0, 1, 0], [1, 1, 1]])
        >>> geom.get_best_plane_from_points_from_two_segments(segment_1, segment_2)
        (array([-0.45440135, -0.45440135,  0.76618459,  0.2628552 ]),
        0.15692966918274637)

    '''
    points = np.array([segment_1[0], segment_1[1], segment_2[0], segment_2[1]])
    a, b, c, d, sse = fit_plane_svd(points)
    best_plane = np.array([a, b, c, d])
    return best_plane, sse

def get_point_of_plane_closest_to_given_point(plane: np.ndarray, point: np.ndarray) -> np.ndarray:
    '''
    Get the point of a plane closest to a given point.

    :param plane: Plane described as Ax + By + Cz + D = 0.
    :type plane: np.ndarray
    :param point: Point.
    :type point: np.ndarray
    :return: Point of plane closest to given point.
    :rtype: np.ndarray

    :Example:
    '''
    # Step 1: Calculate the normal vector of the plane.
    normal = plane[:3]
    # Step 2: Calculate the point of the plane closest to the given point.
    t = - (np.dot(normal, point) + plane[3]) / np.dot(normal, normal)
    closest_point = point + t * normal
    return closest_point

def get_centroid_of_points(points: np.ndarray) -> np.ndarray:
    '''
    Get the centroid of a set of points.

    :param points: Points.
    :type points: np.ndarray
    :return: Centroid.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import mrdja.geometry as geom
        >>> import numpy as np
        >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        >>> centroid = geom.get_centroid_of_points(points)
        >>> centroid
        array([0.33333333, 0.33333333, 0.        ])

    '''
    return np.mean(points, axis=0)

def get_a_polygon_from_plane_equation_and_point(plane: np.ndarray, point: np.ndarray, scale: float = 1.0)->np.ndarray:
    '''
    Get a polygon from plane equation and point.

    :param plane: Plane equation coefficients.
    :type plane: np.ndarray
    :param point: Point.
    :type point: np.ndarray
    :param scale: Scale.
    :type scale: float
    :return: Polygon.
    :rtype: np.ndarray

    :Example:

    ::

        >>> import mrdja.geometry as geom
        >>> import numpy as np
        >>> plane = np.array([0, 0, 1, -3])
        >>> point = np.array([1, 1, 1])
        >>> polygon = geom.get_a_polygon_from_plane_equation_and_point(plane, point)
        >>> polygon
    '''
    # Step 1: Calculate the normal vector of the plane.
    normal = plane[:3]
    # Step 2: Calculate the point of the plane closest to the given point.
    t = - (np.dot(normal, point) + plane[3]) / np.dot(normal, normal)
    closest_point = point + t * normal
    # Step 3: Calculate two perpendicular unit vectors in the plane.
    perpendicular1, perpendicular2 = get_two_perpendicular_unit_vectors_in_plane(plane)
    # Step 4: Calculate the vertices of the polygon.
    vertex1 = closest_point + perpendicular1 * scale + perpendicular2 * scale
    vertex2 = closest_point - perpendicular1 * scale + perpendicular2 * scale
    vertex3 = closest_point - perpendicular1 * scale - perpendicular2 * scale
    vertex4 = closest_point + perpendicular1 * scale - perpendicular2 * scale
    # Step 5: Return the polygon.
    return np.array([vertex1, vertex2, vertex3, vertex4])
