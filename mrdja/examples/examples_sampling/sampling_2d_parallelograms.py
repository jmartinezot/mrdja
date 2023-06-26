import mrdja.sampling as sampling
import matplotlib.pyplot as plt
import numpy as np
import mrdja.geometry as geometry

def get_parallelogram_vertices(center, normal1, normal2, length1, length2):
    vertex1 = center + normal1 * length1 / 2 + normal2 * length2 / 2
    vertex2 = center - normal1 * length1 / 2 + normal2 * length2 / 2
    vertex3 = center - normal1 * length1 / 2 - normal2 * length2 / 2
    vertex4 = center + normal1 * length1 / 2 - normal2 * length2 / 2
    return [vertex1, vertex2, vertex3, vertex4]

n_samples = 100
normal1 = (1, 1)
normal2 = (-1, 1)
center = (0, 0)
length1 = 5
length2 = 5
normal1 = np.array(normal1)
normal2 = np.array(normal2)
center = np.array(center)
samples = sampling.sampling_parallelogram_2d(n_samples, normal1, normal2, center, length1, length2)
fig, ax = plt.subplots()
ax.scatter(*zip(*samples))
ax.set_aspect('equal')
# create title from n_samples, center, and radius, usign fstring
ax.set_title(f'{n_samples} Samples on a Parallelogram with normal vectors {normal1} and {normal2}, center {center}, length1 {length1}, and length2 {length2}')

# Draw the parallelogram
vertices = geometry.get_parallelogram_2d_vertices(center, normal1, normal2, length1, length2)
vertices.append(vertices[0])  # Connect the last vertex with the first to close the shape
vertices = np.array(vertices)
ax.plot(vertices[:, 0], vertices[:, 1], color='r')

ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()
