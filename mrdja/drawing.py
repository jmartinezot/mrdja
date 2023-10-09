import numpy as np
import open3d as o3d
import mrdja.geometry as geom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_plane_as_lines_open3d(A, B, C, D, size=10, line_color=[1, 0, 0]):
    # Define the vertices of the plane
    vertices = np.array([
        [-size, -size, -(D + A * -size + B * -size) / C],
        [size, -size, -(D + A * size + B * -size) / C],
        [size, size, -(D + A * size + B * size) / C],
        [-size, size, -(D + A * -size + B * size) / C]
    ])

    # Define the lines
    lines = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ])

    # Create the LineSet
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(lines)

    # Set the color for each line
    colors = [line_color for i in range(len(lines))]
    lineset.colors = o3d.utility.Vector3dVector(colors)

    return lineset

def draw_plane_as_lines_open3d(A, B, C, D, size=10, line_color=[1, 0, 0], grid_density=5, external_point=None):
    vertices = []
    lines = []
    plane = np.array([A, B, C, D])
    
    if external_point is not None:
        center = geom.get_point_of_plane_closest_to_given_point(plane, external_point)
        center_x, center_y, center_z = center[0], center[1], center[2]
    else:
        center_x, center_y, center_z = 0, 0, 0
    
    # Add the corners of the grid
    for i in np.linspace(-size, size, grid_density):
        for j in np.linspace(-size, size, grid_density):
            x = center_x + i
            y = center_y + j
            z = -(D + A * x + B * y) / C
            vertices.append([x, y, z])

    n = len(vertices)
    vertices = np.array(vertices)
    
    # Add the lines for the grid (horizontal and vertical)
    for i in range(grid_density):
        for j in range(grid_density - 1):
            # horizontal lines
            lines.append([i * grid_density + j, i * grid_density + j + 1])
            # vertical lines
            lines.append([j * grid_density + i, (j + 1) * grid_density + i])

    lines = np.array(lines)
    
    # Create the LineSet
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    
    # Set the color for each line
    colors = [line_color for _ in range(len(lines))]
    lineset.colors = o3d.utility.Vector3dVector(colors)

    return lineset




def draw_face_of_cube(plane: np.ndarray, cube_min: np.ndarray, cube_max: np.ndarray, color: str = "red", alpha: float = 0.5, ax = None):
    '''
    Draws a plane in 3D space using matplotlib.

    :param plane: A 4x1 numpy array containing the coefficients of the plane corresponding to the face.
    :type plane: numpy.ndarray
    :param cube_min: A 3x1 numpy array containing the minimum x, y, and z values of the cube.
    :type cube_min: numpy.ndarray
    :param cube_max: A 3x1 numpy array containing the maximum x, y, and z values of the cube.
    :type cube_max: numpy.ndarray
    :param color: The color of the plane.
    :type color: str
    :param alpha: The transparency of the plane.
    :type alpha: float
    :param ax: The matplotlib axis to draw on.
    :type ax: matplotlib.axes.Axes3D
    :return: None
    :rtype: None
    '''
    plot_inside_function = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_inside_function = True
    # Define the plane's normal vector
    normal = plane[:3]
    # Create a grid of points on the plane for visualization
    x_vals = np.linspace(cube_min[0], cube_max[0], 50)
    y_vals = np.linspace(cube_min[1], cube_max[1], 50)
    z_vals = np.linspace(cube_min[2], cube_max[2], 50)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    if normal[2] != 0:
        z_grid = (-normal[0] * x_grid - normal[1] * y_grid - plane[3]) / normal[2]
    elif normal[1] != 0:
        x_grid, z_grid = np.meshgrid(x_vals, z_vals)
        y_grid = (-normal[0] * x_grid - normal[2] * z_grid - plane[3]) / normal[1]
    else:
        y_grid, z_grid = np.meshgrid(y_vals, z_vals)
        x_grid = (-normal[1] * y_grid - normal[2] * z_grid - plane[3]) / normal[0]
    # Plot the plane
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, color=color)
    if plot_inside_function:
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Show the plot
        plt.show()
    else:
        return ax

def draw_cube(cube_min: np.ndarray, cube_max: np.ndarray, color: str = "red", alpha: float = 0.5, ax = None):
    '''
    Draws a cube in 3D space using matplotlib.

    :param cube_min: A 3x1 numpy array containing the minimum x, y, and z values of the cube.
    :type cube_min: numpy.ndarray
    :param cube_max: A 3x1 numpy array containing the maximum x, y, and z values of the cube.
    :type cube_max: numpy.ndarray
    :param color: The color of the cube.
    :type color: str
    :param alpha: The transparency of the cube.
    :type alpha: float
    :return: None
    :rtype: None

    :Example:

    ::

        >>> import mrdja.drawing as drawing
        >>> import numpy as np
        >>> cube_min = np.array([-2, -2, -1])
        >>> cube_max = np.array([1, 2, 2])
        >>> drawing.draw_cube(cube_min, cube_max, color="red", alpha=0.5)

    |drawing_draw_cube_example|

    .. |drawing_draw_cube_example| image:: ../../_static/images/drawing_draw_cube_example.png

    '''
    plot_inside_function = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_inside_function = True
    # Plane equations for the cube faces (front, back, top, bottom, left, right).
    planes = [
        (1, 0, 0, -cube_min[0]),  # Front face
        (-1, 0, 0, cube_max[0]),  # Back face
        (0, 1, 0, -cube_min[1]),  # Top face
        (0, -1, 0, cube_max[1]),  # Bottom face
        (0, 0, 1, -cube_min[2]),  # Left face
        (0, 0, -1, cube_max[2])   # Right face
    ]
    # Draw the cube faces
    for plane in planes:
        # draw a plane limited to the cube's bounds
        draw_face_of_cube(plane, cube_min, cube_max, color=color, alpha=alpha, ax=ax)
    if plot_inside_function:
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Show the plot
        plt.show()
    else:
        return ax

def draw_line_extension_to_plane(line: np.ndarray, plane: np.ndarray, ax = None):
    '''
    Draws a line in 3D space, and the intersection point of the line and the plane, but this does not draw the plane.

    :param line: A 2x3 numpy array containing the endpoints of the line.
    :type line: numpy.ndarray
    :param plane: A 4x1 numpy array containing the coefficients of the plane.
    :type plane: numpy.ndarray
    :return: None
    :rtype: None

    :Example:

    ::

        >>> import mrdja.geometry as geom
        >>> import numpy as np 
        >>> import mrdja.drawing as drawing
        >>> import matplotlib.pyplot as plt
        >>> line = np.array([[0, 0, 0], [1, 1, 1]])
        >>> plane = np.array([0, 0, 1, -3])
        >>> intersection_point = geom.get_intersection_point_of_line_with_plane(line, plane)
        >>> intersection_point
        array([3., 3., 3.])
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> drawing.draw_line_extension_to_plane(line, plane, ax)
        >>> 
        >>> drawing.draw_line_extension_to_plane(line, plane)

    |drawing_draw_line_extension_to_plane_example|

    .. |drawing_draw_line_extension_to_plane_example| image:: ../../_static/images/drawing_draw_line_extension_to_plane_example.png

    '''
    plot_inside_function = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_inside_function = True
    intersection_point = geom.get_intersection_point_of_line_with_plane(line, plane)
    # Calculate the distances from line[0] and line[1] to the intersection point
    distance_to_line0 = np.linalg.norm(intersection_point - line[0])
    distance_to_line1 = np.linalg.norm(intersection_point - line[1])

    # Determine which point is farther away
    farther_point_index = 0 if distance_to_line0 > distance_to_line1 else 1
    farther_point = line[farther_point_index]

    # Plot the remaining portion of the line from the farther point to the intersection point
    intersection_x = intersection_point[0]
    intersection_y = intersection_point[1]
    intersection_z = intersection_point[2]
    farther_x = farther_point[0]
    farther_y = farther_point[1]
    farther_z = farther_point[2]
    ax.plot([farther_x, intersection_x], [farther_y, intersection_y], [farther_z, intersection_z], linestyle='--', color='blue')

    # Plot the intersection point
    ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2], c='green', label='Intersection Point', s=100)

    if plot_inside_function:
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Show the plot
        plt.show()
    else:
        return ax

