import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a1 = 1.086
a2 = 3.084
a3 = 2.118
b1 = -1.274
b2 = -0.005
b3 = 0.198

# Define vectors A and B
A = np.array([a1, a2, a3])  # Replace with your values
B = np.array([b1, b2, b3])  # Replace with your values

# Create a meshgrid for plotting the plane defined by A and B
x = np.linspace(-10, 10, 10)
y = np.linspace(-10, 10, 10)
X, Y = np.meshgrid(x, y)

# Calculate the normal of the plane
normal = np.cross(A, B)

# Find a vector C perpendicular to A and inside the plane defined by A and B
C = np.cross(normal, A)

# normalize A, B and C
A = A / np.linalg.norm(A)
B = B / np.linalg.norm(B)
C = C / np.linalg.norm(C)

# Plot A, B , D and the plane
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.quiver(0, 0, 0, A[0], A[1], A[2], color='g')
ax.quiver(0, 0, 0, B[0], B[1], B[2], color='g')
ax.quiver(0, 0, 0, C[0], C[1], C[2], color='b')
ax.quiver(0, 0, 0, normal[0], normal[1], normal[2], color='r')
ax.plot_surface(X, Y, (normal[0] * X + normal[1] * Y) / (-normal[2]), alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()




