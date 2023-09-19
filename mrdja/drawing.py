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

def extend_line_to_cube_limits(line, cube_min, cube_max):
    # Calculate the direction vector of the line.
    direction_vector = line[1] - line[0]

    # Find the intersection points of the line with the cube walls.
    intersection_points = []
    for i in range(3):
        # Check if the line is parallel to the cube wall.
        if direction_vector[i] == 0:
            continue

        # Check if the line intersects the cube wall at the minimum point.
        t_min = (cube_min[i] - line[0][i]) / direction_vector[i]
        if t_min >= 0 and t_min <= 1:
            intersection_point = line[0] + t_min * direction_vector
            if np.all(intersection_point >= cube_min) and np.all(intersection_point <= cube_max):
                intersection_points.append(intersection_point)

        # Check if the line intersects the cube wall at the maximum point.
        t_max = (cube_max[i] - line[0][i]) / direction_vector[i]
        if t_max >= 0 and t_max <= 1:
            intersection_point = line[0] + t_max * direction_vector
            if np.all(intersection_point >= cube_min) and np.all(intersection_point <= cube_max):
                intersection_points.append(intersection_point)

    # Extend the line to the intersection points.
    extended_line_endpoints = np.vstack((line[0], intersection_points[0]))
    if len(intersection_points) > 1:
        extended_line_endpoints = np.vstack((extended_line_endpoints, intersection_points[1]))

    return extended_line_endpoints

def draw_line_extension_to_plane(line, plane):
    intersection_point = geom.get_intersection_point_of_line_with_plane(line, plane)
    # Calculate the distances from line[0] and line[1] to the intersection point
    distance_to_line0 = np.linalg.norm(intersection_point - line[0])
    distance_to_line1 = np.linalg.norm(intersection_point - line[1])

    # Determine which point is farther away
    farther_point_index = 0 if distance_to_line0 > distance_to_line1 else 1
    farther_point = line[farther_point_index]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the remaining portion of the line from the farther point to the intersection point
    intersection_x = intersection_point[0]
    intersection_y = intersection_point[1]
    intersection_z = intersection_point[2]
    farther_x = farther_point[0]
    farther_y = farther_point[1]
    farther_z = farther_point[2]
    ax.plot([farther_x, intersection_x], [farther_y, intersection_y], [farther_z, intersection_z], linestyle='--', color='blue')

    # Define the plane's normal vector
    normal = plane[:3]

    # Create a grid of points on the plane for visualization
    x_vals = np.linspace(-5, 5, 50)
    y_vals = np.linspace(-5, 5, 50)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    z_grid = (-normal[0] * x_grid - normal[1] * y_grid - plane[3]) / normal[2]

    # Plot the plane
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, label='Plane', color='red')

    # Plot the intersection point
    ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2], c='green', label='Intersection Point', s=100)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()
    return

