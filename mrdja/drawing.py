import numpy as np
import open3d as o3d

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
