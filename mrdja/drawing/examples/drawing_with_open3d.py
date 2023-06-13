import open3d as o3d
import numpy as np
import mrdja.geometry as geometry
import mrdja.sampling as sampling
import mrdja.drawing as drawing
import mrdja.ransac.coreransac as coreransac
import mrdja.ransac as ransac
from sklearn.cluster import DBSCAN, KMeans

# Create a 3D parallelogram
n_samples = 10000
normal1 = (1, 0, 0)
normal2 = (0, 1, 0)
normal3 = (0, 0, 1)
center = (0, 0, 0)
length1 = 10
length2 = 10
length3 = 0.5
normal1 = np.array(normal1)
normal2 = np.array(normal2)
normal3 = np.array(normal3)
center = np.array(center)
samples = sampling.sampling_parallelogram_3d(n_samples, normal1, normal2, normal3, center, length1, length2, length3)
samples = np.array(samples)

# Create a 3D point cloud
pcd_plane = o3d.geometry.PointCloud()
pcd_plane.points = o3d.utility.Vector3dVector(samples)
pcd_plane.paint_uniform_color([0, 1, 0])

# Get the plane equation
plane_equation = geometry.get_plane_equation(normal1, normal2, center)

sphere_radius = 10
sphere_center = (0, 0, 0)
sphere_points = sampling.sampling_sphere(n_samples=100, center=sphere_center, radius=sphere_radius, seed=None)
# Create a 3D point cloud
pcd_sphere_points = o3d.geometry.PointCloud()
pcd_sphere_points.points = o3d.utility.Vector3dVector(sphere_points)
pcd_sphere_points.paint_uniform_color([1, 0, 0])

mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=20) # created at origin
mesh_sphere = mesh_sphere.translate(sphere_center)
pcd_sphere_boundary = mesh_sphere.sample_points_poisson_disk(number_of_points=100, init_factor=5)
pcd_sphere_boundary.paint_uniform_color([1, 0, 0])

# Draw the point cloud, the plane and the sphere
true_plane_geometry = drawing.draw_plane_as_lines_open3d(*plane_equation, line_color=[0, 0, 1])
o3d.visualization.draw_geometries([pcd_plane, pcd_sphere_points, pcd_sphere_boundary, true_plane_geometry])
list_geometries = [pcd_plane, pcd_sphere_points, pcd_sphere_boundary, true_plane_geometry]

pcd = pcd_plane + pcd_sphere_points + pcd_sphere_boundary

len_points = len(pcd.points)
for i in range(20):
    ransac_results = coreransac.get_ransac_iteration_results(np.asanyarray(pcd.points), len_points, threshold=0.5)
    print(ransac_results)
    current_plane = ransac_results["current_plane"]
    list_geometries.append(drawing.draw_plane_as_lines_open3d(*current_plane, line_color=[0, 1, 1]))
o3d.visualization.draw_geometries(list_geometries)

# get 20 random points from the pcd_plane point cloud
random_points = sampling.sampling_pcd_points(pcd_plane, 200)
closest_plane = geometry.find_closest_plane(random_points)
# draw the closest plane along with the random points
closest_plane_geometry = drawing.draw_plane_as_lines_open3d(*closest_plane, line_color=[0, 1, 1])
o3d.visualization.draw_geometries([pcd_plane, closest_plane_geometry, true_plane_geometry])

k = 500
dict_distances = ransac.generate_k_planes_from_k_sets_of_3_random_points_of_pointcloud_points_and_compute_the_distance_from_the_planes_to_all_the_points(pcd, k)
# distances is a dictionary of the form {plane: distances}
# where plane is a tuple of the form (A, B, C, D) and distances is a np.array of shape (len(pcd.points),)
# apply KMEANS clustering to the distances
distances = np.array(list(dict_distances.values()))
planes = np.array(list(dict_distances.keys()))
kmeans = KMeans(n_clusters=5, random_state=0).fit(distances)
# get the labels
labels = kmeans.labels_
# get the number of clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# get the indices of the points that belong to each cluster
clusters_indices = [np.where(labels == i)[0] for i in range(n_clusters)]
# get the planes that belong to each cluster
clusters_planes = []
for indices_list in clusters_indices:
    print(indices_list)
    clusters_planes.append(planes[indices_list])
# plot the planes that belong to each cluster in a different color
list_geometries = [pcd_plane, true_plane_geometry]
for cluster_plane in clusters_planes:
    line_color = np.random.rand(3)
    for plane in cluster_plane:
        list_geometries.append(drawing.draw_plane_as_lines_open3d(*plane, line_color=line_color))
o3d.visualization.draw_geometries(list_geometries)
# order the clusters_planes by the number of planes they contain
clusters_planes = sorted(clusters_planes, key=lambda x: len(x), reverse=True)
list_geometries = [pcd_plane, true_plane_geometry]
for plane in clusters_planes[0]:
    list_geometries.append(drawing.draw_plane_as_lines_open3d(*plane, line_color=[0, 1, 1]))
o3d.visualization.draw_geometries(list_geometries)
