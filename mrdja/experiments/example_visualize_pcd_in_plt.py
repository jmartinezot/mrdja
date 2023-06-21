# visualize a point cloud in matplotlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import mrdja.ransac.coreransac as coreransac

filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/conferenceRoom_1/conferenceRoom_1.ply"
pcd_all = o3d.io.read_point_cloud(filename)
np_points = np.asarray(pcd_all.points)
max_x = np.max(np_points[:, 0])
min_x = np.min(np_points[:, 0])
max_y = np.max(np_points[:, 1])
min_y = np.min(np_points[:, 1])
max_z = np.max(np_points[:, 2])
min_z = np.min(np_points[:, 2])
number_pcd_points = len(pcd_all.points)
np_all_points = np.asarray(pcd_all.points)
print("number_pcd_points: ", number_pcd_points)
# downsample the point cloud to make it easier to visualize
pcd = pcd_all.voxel_down_sample(voxel_size=0.1)
number_pcd_points = len(pcd.points)
print("number_pcd_points: ", number_pcd_points)
np_points = np.asarray(pcd.points)
np_colors = np.asarray(pcd.colors)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Create a meshgrid for plotting the plane defined by A and B
x = np.linspace(min_x-1, max_x+1, 10)
y = np.linspace(min_y-1, max_y+1, 10)
X, Y = np.meshgrid(x, y)

np.random.seed(42)
for i in range(0, 30):
# perform a ransac iteration to find a plane
    dict_results = coreransac.get_ransac_iteration_results(np_all_points, threshold=0.05)
    current_plane = dict_results["current_plane"]
    print("current_plane: ", current_plane)
    A, B, C, D = current_plane
    ax.plot_surface(X, Y, (-(A * X + B * Y + D) / C), alpha=0.01)
ax.scatter(np_points[:, 0], np_points[:, 1], np_points[:, 2], c=np_colors)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# set axis limits to make the plot look better. The limits are based on the point cloud
ax.set_xlim(min_x-1, max_x+1)
ax.set_ylim(min_y-1, max_y+1)
ax.set_zlim(min_z-1, max_z+1)
plt.show()