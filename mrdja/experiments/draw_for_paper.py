import mrdja.sampling as sampling
import mrdja.geometry as geom
import mrdja.drawing as drawing
import mrdja.matplot3d as plt3d
import matplotlib.pyplot as plt
import numpy as np

n_samples = 5
normal1 = (1, 0, 0)
normal2 = (0, 1, 0)
center = (.5, .5, 0)
length1 = 1
length2 = 1

points_plane = sampling.sampling_parallelogram_3d(n_samples=n_samples, normal1=normal1, normal2=normal2, 
                                          center=center, length1=length1, length2=length2, seed=43)
points_plane = np.array(points_plane)

points_plane = np.array([[0.8, 0.2, 0], [0.1, 0.8, 0], [0.2, 0.25, 0], [0.6, 0.75, 0], [0.5, 0.5, 0], [0.2, 0.95, 0]])
points_outside_plane = np.array([[0, 0.8, 0.6], [0.6, 1, 0.8], [0.9, 1, 0.6], [0.9, 1, 0.2]])

plane = np.array([0, 0, 1, 0])
cube_min = np.array([0, 0, 0])
cube_max = np.array([1, 1, 1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

color_plane = "green"
color_good_points = "green"
color_bad_points = "red"
color_good_lines = "green"
color_bad_lines = "red"
alpha = 0.1
polygon_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
plt3d.draw_polygon(polygon_points, color=color_plane, alpha=alpha, ax=ax)
plt3d.draw_points(points_plane, color=color_good_points, style="o", ax=ax)
plt3d.draw_points(points_outside_plane, color=color_bad_points, style="o", ax=ax)

# save the figure
plt.savefig("plane.png")

good_segment_1 = np.array([points_plane[0], points_plane[2]])
good_segment_2 = np.array([points_plane[4], points_plane[1]])

plt3d.draw_segment(good_segment_1, color=color_good_lines, ax=ax)
plt3d.draw_segment(good_segment_2, color=color_good_lines, ax=ax)

bad_segment_1 = np.array([points_plane[3], points_outside_plane[1]])
bad_segment_2 = np.array([points_plane[5], points_outside_plane[0]])
bad_segment_3 = np.array([points_outside_plane[2], points_outside_plane[3]])

plt3d.draw_segment(bad_segment_1, color=color_bad_lines, ax=ax)
plt3d.draw_segment(bad_segment_2, color=color_bad_lines, ax=ax)
plt3d.draw_segment(bad_segment_3, color=color_bad_lines, ax=ax)

# save the figure
plt.savefig("lines.png")

bad_plane = geom.get_best_plane_from_points_from_two_segments(bad_segment_1, bad_segment_2)[0]
perpendicular1, perpendicular2 = geom.get_two_perpendicular_unit_vectors_in_plane(plane)
points_of_bad_segments = np.vstack((bad_segment_1, bad_segment_2))
bad_centroid = geom.get_centroid_of_points(points_of_bad_segments)
bad_polygon = geom.get_a_polygon_from_plane_equation_and_point(bad_plane, bad_centroid, scale = 0.5)

plt3d.draw_polygon(bad_polygon, color="red", alpha=alpha, ax=ax)

# save the figure
plt.savefig("bad_polygon.png")

plt.show()