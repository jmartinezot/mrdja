import random
import math
import numpy as np

def sample_point_circle_2d(radius=1):
    while True:
        x = random.uniform(-radius, radius)
        y = random.uniform(-radius, radius)
        if x**2 + y**2 <= radius**2:
            return x, y

def sampling_circle_2d(n_samples, radius=1):
    """
    Generate random samples on a 2D circle with a specified radius.

    :param n_samples: int
        The number of random samples to generate on the circle.
    :param radius: float, optional
        The radius of the circle. Default is 1.
    :return: list of tuple
        A list of tuples (x, y) representing the coordinates of the sampled points.
        
    .. note:: The samples are uniformly distributed on the circle.
    """
    samples = [sample_point_circle_2d(radius) for _ in range(n_samples)]
    return samples

def sample_point_circle_3d_rejection(radius=1, center=np.array([0, 0, 0]), normal=np.array([0, 0, 1])):
    normal = normal / np.linalg.norm(normal)
    
    while True:
        # Generate a random point within the bounding box
        point = center + np.array([random.uniform(-radius, radius) for _ in range(3)])

        # Project the point onto the plane defined by the circle
        projected_point = point - np.dot(point - center, normal) * normal

        # Check if the projected point lies within the circle
        if np.linalg.norm(projected_point - center) <= radius:
            return projected_point

def sampling_circle_3d_rejection(n_samples, radius=1, center=np.array([0, 0, 0]), normal=np.array([0, 0, 1])):
    samples = [sample_point_circle_3d_rejection(radius, center, normal) for _ in range(n_samples)]
    return samples

def sample_point_parallelogram(a, b, c):
    u = random.random()
    v = random.random()
    x = a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0])
    y = a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1])
    return x, y

def sampling_parallelogram_2d(n_samples, a, b, c):
    samples = [sample_point_parallelogram(a, b, c) for _ in range(n_samples)]
    return samples

def sample_point_cuboid(a, b, c, d, h):
    u = random.random()
    v = random.random()
    w = random.random()
    x = a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0]) + w * (d[0] - a[0])
    y = a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1]) + w * (d[1] - a[1])
    z = a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2]) + w * (d[2] - a[2]) + h
    return x, y, z

def sampling_cuboid(n_samples, a, b, c, d, h):
    samples = [sample_point_cuboid(a, b, c, d, h) for _ in range(n_samples)]
    return samples

def sample_point_sphere(radius=1):
    while True:
        x = random.uniform(-radius, radius)
        y = random.uniform(-radius, radius)
        z = random.uniform(-radius, radius)
        if x**2 + y**2 + z**2 <= radius**2:
            return x, y, z

def sampling_sphere(n_samples, radius=1):
    samples = [sample_point_sphere(radius) for _ in range(n_samples)]
    return samples


