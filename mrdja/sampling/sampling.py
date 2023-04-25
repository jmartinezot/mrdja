import random
import math
import numpy as np

def set_random_seed(seed):
    """
    Set the random seed for reproducibility.

    :param seed: int
        The seed value for the random number generator.
    """
    random.seed(seed)

def sample_point_circle_2d(center, radius=1):
    """
    Generate a random point on a 2D circle with a specified radius and center.

    :param center: tuple
        A tuple (x, y) representing the center of the circle.
    :param radius: float, optional
        The radius of the circle. Default is 1.
    :return: tuple
        A tuple (x, y) representing the coordinates of the sampled point.
    """
    while True:
        # Generate a random point on the circle with radius 1 centered at (0,0)
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        
        # Scale the coordinates to the desired radius
        x *= radius
        y *= radius
        
        # Translate the point to the desired center
        x += center[0]
        y += center[1]
        
        # Check if the point is within the circle
        if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
            return x, y

def sampling_circle_2d(n_samples, center=(0,0), radius=1, seed=None):
    """
    Generate random samples on a 2D circle with a specified radius and center.

    :param n_samples: int
        The number of random samples to generate on the circle.
    :param center: tuple, optional
        A tuple (x, y) representing the center of the circle. Default is (0,0).
    :param radius: float, optional
        The radius of the circle. Default is 1.
    :param seed: int, optional
        The seed value for the random number generator. Default is None.
    :return: list of tuple
        A list of tuples (x, y) representing the coordinates of the sampled points.
        
    .. note:: The samples are uniformly distributed on the circle.
    
    Examples
    --------
    Generate 100 random samples on a circle with center (2,3) and radius 5:
    
    >>> samples = sampling_circle_2d(100, center=(2,3), radius=5)
    
    Plot the samples:
    
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.scatter(*zip(*samples))
    >>> ax.set_aspect('equal')
    >>> ax.set_title('100 Samples on a Circle with Center (2,3) and Radius 5')
    >>> ax.set_xlabel('X')
    >>> ax.set_ylabel('Y')
    >>> plt.show()
    
    .. image:: ../../../images/samples1.png
    
    Generate 10000 random samples on a circle with center (0,0) and radius 2:
    
    >>> samples = sampling_circle_2d(10000, radius=2)
    
    Compute the sample mean and standard deviation:
    
    >>> import numpy as np
    >>> mean = np.mean(samples, axis=0)
    >>> std = np.std(samples, axis=0)
    
    Print the results:
    
    >>> print(f"Sample mean: {mean}")
    >>> print(f"Sample standard deviation: {std}")
    
    Create a plot:
    
    >>> fig, ax = plt.subplots()
    >>> ax.scatter(*zip(*samples))
    >>> ax.set_aspect('equal')
    >>> ax.set_title('10000 Samples on a Circle with Center (0,0) and Radius 2')
    >>> ax.set_xlabel('X')
    >>> ax.set_ylabel('Y')
    >>> plt.show()
    
    .. image:: ../../../images/samples2.png
    """
    # Set the random seed for reproducibility if provided
    if seed is not None:
        set_random_seed(seed)

    # Generate the specified number of samples on the circle
    samples = [sample_point_circle_2d(center, radius) for _ in range(n_samples)]

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


