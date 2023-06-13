import mrdja.sampling as sampling
import matplotlib.pyplot as plt
import numpy as np
import mrdja.geometry as geometry
from mpl_toolkits.mplot3d import Axes3D

n_samples = 100
normal1 = (1, 1, 0)
normal2 = (0, 1, 0)
normal3 = (-1, 0, 1)
center = (1, 0, 1)
length1 = 5
length2 = 5
length3 = 5
normal1 = np.array(normal1)
normal2 = np.array(normal2)
normal3 = np.array(normal3)
center = np.array(center)
samples = sampling.sampling_parallelogram_3d(n_samples, normal1, normal2, normal3, center, length1, length2, length3)
samples = np.array(samples)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])


# create title from n_samples, center, and radius, using fstring
ax.set_title(f'{n_samples} Samples on a Parallelogram with normal vectors {normal1}, {normal2} and {normal3}, \
            center {center}, length1 {length1}, length2 {length2} and length3 {length3}')

# Draw the parallelogram
vertices = geometry.get_parallelogram_3d_vertices(center, normal1, normal2, normal3, length1, length2, length3)
# Define the edges of the 3d parallelogram
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
# Plot the edges
for edge in edges:
    ax.plot([vertices[edge[0]][0], vertices[edge[1]][0]],
            [vertices[edge[0]][1], vertices[edge[1]][1]],
            [vertices[edge[0]][2], vertices[edge[1]][2]], color='red')
    
# Draw the normals at a quarter of their corresponding length
quarter_length1 = length1 / 4
quarter_length2 = length2 / 4
arrow_length1 = quarter_length1 / 2
arrow_length2 = quarter_length2 / 2
ax.quiver(center[0], center[1], center[2], normal1[0], normal1[1], normal1[2], length=arrow_length1, normalize=False, color='red')
ax.quiver(center[0], center[1], center[2], normal2[0], normal2[1], normal2[2], length=arrow_length2, normalize=False, color='red')
ax.quiver(center[0], center[1], center[2], normal3[0], normal3[1], normal3[2], length=arrow_length2, normalize=False, color='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
