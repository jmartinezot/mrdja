import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate random points in 3D space
num_points = 100
np.random.seed(42)
points = np.random.rand(num_points, 3)

# Define the function for fitting a plane using SVD
def fit_plane(points):
    centroid = np.mean(points, axis=0)
    shifted_points = points - centroid
    _, _, V = np.linalg.svd(shifted_points)
    normal = V[2]
    d = -np.dot(normal, centroid)
    return normal, d

# Fit the plane to the points
normal, d = fit_plane(points)

# Create a grid of points to represent the plane
xx, yy = np.meshgrid(np.linspace(-1, 2, 10), np.linspace(-1, 2, 10))
zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]

# Plot the points and the fitted plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
ax.plot_surface(xx, yy, zz, alpha=0.2)

# Set labels and show the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
