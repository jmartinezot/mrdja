'''
Module for functions that take an ax object, draw something, and return the ax object. This is useful for chaining calls to these functions.
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mrdja.geometry as geom
import numpy as np

def draw_segment(line: np.ndarray, color: str = 'red', style: str = '-', ax = None):
    '''
    Draws a line in 3D space.

    :param segment: A 2x3 numpy array containing the endpoints of the segment.
    :type segment: numpy.ndarray
    :param color: The color of the segment to draw.
    :type color: str
    :param style: The style of the segment to draw.
    :type style: str
    :param ax: The matplotlib ax object to draw on.
    :type ax: matplotlib.axes._subplots.Axes3DSubplot
    :return: The ax object.
    :rtype: matplotlib.axes._subplots.Axes3DSubplot

    :Example:

    ::

    >>> import mrdja.matplot3d as plt3d
    >>> import numpy as np
    >>> segment = np.array([[0, 0, 0], [1, 1, 1]])
    >>> plt3d.draw_segment(segment, color="blue", style="--")
    
    |matplot3d_draw_segment_example|

    .. |matplot3d_draw_segment_example| image:: ../../_static/images/matplot3d_draw_segment_example.png
    '''
    plot_inside_function = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_inside_function = True
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color=color, linestyle=style)
    if plot_inside_function:
        plt.show()
    else:
        return ax
    

def draw_polygon(vertices: np.ndarray, color: str = 'red', alpha: float = 0.5, ax = None):
    '''
    Draws a polygon in 3D space.

    :param vertices: A Nx3 numpy array containing the vertices of the polygon.
    :type vertices: numpy.ndarray
    :param color: The color of the polygon to draw.
    :type color: str
    :param style: The style of the polygon to draw.
    :type style: str
    :param ax: The matplotlib ax object to draw on.
    :type ax: matplotlib.axes._subplots.Axes3DSubplot
    :return: The ax object.
    :rtype: matplotlib.axes._subplots.Axes3DSubplot

    :Example:

    ::

    >>> import mrdja.matplot3d as plt3d
    >>> import numpy as np
    >>> vertices = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0]])
    >>> plt3d.draw_polygon(vertices, color="blue", alpha=0.2)
    
    |matplot3d_draw_polygon_example|

    .. |matplot3d_draw_polygon_example| image:: ../../_static/images/matplot3d_draw_polygon_example.png
    '''
    plot_inside_function = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_inside_function = True
    # Create a collection of polygons
    polygon = [vertices]
    # Plot the polygon
    ax.add_collection3d(Poly3DCollection(polygon, facecolors=color, alpha=alpha))

    if plot_inside_function:
        plt.show()
    else:
        return ax
    
def draw_circle(center: np.ndarray = np.array([0, 0, 0]), radius: float = 1, normal: np.ndarray = np.array([0, 0 ,1]), color: str = 'red', alpha: float = 0.5, ax = None):
    '''
    Draws a circle in 3D space.

    :param center: A 3x1 numpy array containing the center of the circle.
    :type center: numpy.ndarray
    :param radius: The radius of the circle to draw.
    :type radius: float
    :param normal: The normal vector of the circle.
    :type normal: numpy.ndarray
    :param color: The color of the circle to draw.
    :type color: str
    :param style: The style of the circle to draw.
    :type style: str
    :param ax: The matplotlib ax object to draw on.
    :type ax: matplotlib.axes._subplots.Axes3DSubplot
    :return: The ax object.
    :rtype: matplotlib.axes._subplots.Axes3DSubplot

    :Example:

    ::

    >>> import mrdja.matplot3d as plt3d
    >>> import numpy as np
    >>> center = np.array([0, 0, 0])
    >>> radius = 1
    >>> normal = np.array([0.5, 0.3, 0.8]) / np.linalg.norm(np.array([0.5, 0.3, 0.8]))
    >>> plt3d.draw_circle(center, radius, normal, color="blue", alpha=0.2)

    |matplot3d_draw_circle_example|

    .. |matplot3d_draw_circle_example| image:: ../../_static/images/matplot3d_draw_circle_example.png

    '''
    plot_inside_function = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_inside_function = True
    # Generate points on the circle's surface
    num_points = 100
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi = np.linspace(0, np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)

    # Calculate an orthogonal vector to the given normal vector
    orthogonal_vector = np.array([1, 0, 0]) if normal[1] == 0 else np.array([0, 1, 0])

    # Calculate the third orthogonal vector
    third_vector = np.cross(normal, orthogonal_vector)

    # Calculate the points on the filled circle
    x = center[0] + radius * (orthogonal_vector[0] * np.sin(phi) * np.cos(theta) + third_vector[0] * np.sin(phi) * np.sin(theta))
    y = center[1] + radius * (orthogonal_vector[1] * np.sin(phi) * np.cos(theta) + third_vector[1] * np.sin(phi) * np.sin(theta))
    z = center[2] + radius * (orthogonal_vector[2] * np.sin(phi) * np.cos(theta) + third_vector[2] * np.sin(phi) * np.sin(theta))

    # Plot the filled circle
    ax.add_collection3d(Poly3DCollection([list(zip(x.flatten(), y.flatten(), z.flatten()))], facecolors=color, alpha=alpha, linewidths=0))

    if plot_inside_function:
        # Compute the minimum and maximum values of x
        min_x = np.min(x)
        max_x = np.max(x)
        # Compute the minimum and maximum values of y
        min_y = np.min(y)
        max_y = np.max(y)
        # Compute the minimum and maximum values of z
        min_z = np.min(z)
        max_z = np.max(z)
        # Compute the limits of the 3D graph
        graph_limits = geom.get_limits_of_3d_graph_from_limits_of_object(min_x, max_x, min_y, max_y, min_z, max_z)
        ax.set_xlim(graph_limits[0], graph_limits[1])  # Set x-axis limits
        ax.set_ylim(graph_limits[2], graph_limits[3])  # Set y-axis limits
        ax.set_zlim(graph_limits[4], graph_limits[5])  # Set z-axis limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    else:
        return ax

def draw_circumference(center: np.ndarray = np.array([0, 0, 0]), radius: float = 1, normal: np.ndarray = np.array([0, 0 ,1]), color: str = 'red', alpha: float = 0.5, ax = None):
    '''
    Draws a circumference in 3D space.

    :param center: A 3x1 numpy array containing the center of the circumference.
    :type center: numpy.ndarray
    :param radius: The radius of the circumference to draw.
    :type radius: float
    :param normal: The normal vector of the circumference.
    :type normal: numpy.ndarray
    :param color: The color of the circumference to draw.
    :type color: str
    :param style: The style of the circumference to draw.
    :type style: str
    :param ax: The matplotlib ax object to draw on.
    :type ax: matplotlib.axes._subplots.Axes3DSubplot
    :return: The ax object.
    :rtype: matplotlib.axes._subplots.Axes3DSubplot

    :Example:

    ::

    >>> import mrdja.matplot3d as plt3d
    >>> import numpy as np
    >>> center = np.array([0, 0, 0])
    >>> radius = 1
    >>> normal = np.array([0.5, 0.3, 0.8]) / np.linalg.norm(np.array([0.5, 0.3, 0.8]))
    >>> plt3d.draw_circumference(center, radius, normal, color="blue", alpha=0.2)

    |matplot3d_draw_circumference_example|

    .. |matplot3d_draw_circumference_example| image:: ../../_static/images/matplot3d_draw_circumference_example.png

    '''
    plot_inside_function = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_inside_function = True
    # Generate points on a 2D circle in the XY plane
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Calculate an orthogonal vector to the normal vector
    orthogonal_vector = np.array([1, 0, 0]) if normal[1] == 0 else np.array([0, 1, 0])

    # Calculate the third orthogonal vector
    third_vector = np.cross(normal, orthogonal_vector)

    # Apply a transformation to rotate and translate the circle
    transformation_matrix = np.column_stack((orthogonal_vector, third_vector, normal))
    transformed_points = np.dot(transformation_matrix, np.vstack((x, y, np.zeros_like(x))))
    translated_points = center[:, np.newaxis] + transformed_points

    # Plot the 3D circle
    ax.plot(translated_points[0], translated_points[1], translated_points[2], color=color, alpha=alpha)

    if plot_inside_function:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    else:
        return ax
    
def draw_points(points: np.ndarray, color: str = 'red', style: str = "o", alpha: float = 0.5, ax = None):
    '''
    Draws points in 3D space.

    :param points: A Nx3 numpy array containing the points to draw.
    :type points: numpy.ndarray
    :param color: The color of the points to draw.
    :type color: str
    :param style: The style of the points to draw.
    :type style: str
    :param alpha: The transparency of the points to draw.
    :type alpha: float
    :param ax: The matplotlib ax object to draw on.
    :type ax: matplotlib.axes._subplots.Axes3DSubplot
    :return: The ax object.
    :rtype: matplotlib.axes._subplots.Axes3DSubplot

    :Example:

    ::

    >>> import mrdja.matplot3d as plt3d
    >>> import numpy as np
    >>> points = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0]])
    >>> plt3d.draw_points(points, color="blue", style="o")


    |matplot3d_draw_points_example|

    .. |matplot3d_draw_points_example| image:: ../../_static/images/matplot3d_draw_points_example.png
    '''
    plot_inside_function = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_inside_function = True
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, marker=style, alpha=alpha)
    if plot_inside_function:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    else:
        return ax
