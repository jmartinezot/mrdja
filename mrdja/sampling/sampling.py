import random
import math
import numpy as np
from typing import List, Tuple

def set_random_seed(seed):
    """
    Set the random seed for reproducibility.

    :param seed: int
        The seed value for the random number generator.
    """
    random.seed(seed)

def sample_point_circle_2d(center, radius=1):
    """
    Generate a random point on a 2D circle with a specified radius and center, using rejection sampling.

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

def sampling_circle_2d(n_samples=1, center=(0,0), radius=1, seed=None):
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

def sample_point_parallelogram_2d(normal1: Tuple[float, float], normal2: Tuple[float, float], center: Tuple[float, float], length1: float, length2: float) -> Tuple[float, float]:
    '''
    Sample a point from a parallelogram with sides parallel to the vectors normal1 and normal2.

    :param normal1: Tuple[float, float]
        The first vector normal to the sides of the parallelogram.
    :param normal2: Tuple[float, float]
        The second vector normal to the sides of the parallelogram.
    :param center: Tuple[float, float]
        The center of the parallelogram.
    :param length1: float
        The length of the first side of the parallelogram.
    :param length2: float
        The length of the second side of the parallelogram.
    :return: Tuple[float, float]
        A tuple (x, y) representing the coordinates of the sampled point.
    '''
    # Generate two random numbers between -length1/2 and length1/2 and -length2/2 and length2/2 respectively
    x = random.uniform(-length1/2, length1/2)
    y = random.uniform(-length2/2, length2/2)
    # Those numbers represent the coordinates of the sampled point in the normal coordinate system
    # Transform the coordinates to the global coordinate system
    projected_point = (x * normal1[0] + y * normal2[0] + center[0], x * normal1[1] + y * normal2[1] + center[1])
    return projected_point

def sampling_parallelogram_2d(n_samples: int, normal1: Tuple[float, float], normal2: Tuple[float, float], center: Tuple[float, float], length1: float, length2: float) -> List[Tuple[float, float]]:
    normal1 = np.array(normal1)
    normal2 = np.array(normal2) 
    center = np.array(center)
    samples = [sample_point_parallelogram_2d(normal1, normal2, center, length1, length2) for _ in range(n_samples)]
    return samples

def sample_point_alligned_parallelogram_2d(min_x: float, max_x: float, min_y: float, max_y: float) -> tuple[float, float]:
    '''
    Sample a point from a parallelogram with sides parallel to the x and y axes.

    :param min_x: float
        The minimum x coordinate of the parallelogram.
    :param max_x: float
        The maximum x coordinate of the parallelogram.
    :param min_y: float
        The minimum y coordinate of the parallelogram.
    :param max_y: float
        The maximum y coordinate of the parallelogram.
    :return: tuple[float, float]
        A tuple (x, y) representing the coordinates of the sampled point.

    .. note:: The samples are uniformly distributed on the parallelogram.

    Examples
    --------
    Sample a point from a parallelogram with sides parallel to the x and y axes:

    >>> sample_point_alligned_parallelogram(-1, 1, -1, 1)
    (-0.5, 0.5)
    '''
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    return x, y

def sampling_alligned_parallelogram_2d(n_samples: int, min_x: float, max_x: float, min_y: float, max_y: float):
    '''
    Sample points from a parallelogram with sides parallel to the x and y axes.

    :param n_samples: int
        The number of samples to generate.
    :param min_x: float
        The minimum x coordinate of the parallelogram.
    :param max_x: float
        The maximum x coordinate of the parallelogram.
    :param min_y: float
        The minimum y coordinate of the parallelogram.
    :param max_y: float
        The maximum y coordinate of the parallelogram.
    :return: list[tuple[float, float]]
        A list of tuples (x, y) representing the coordinates of the sampled points.

    .. note:: The samples are uniformly distributed on the parallelogram.

    Examples
    --------
    Sample 10 points from a parallelogram with sides parallel to the x and y axes:

    >>> sampling_alligned_parallelogram_2d(10, -1, 1, -1, 1)
    [(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, -0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)]
    '''
    samples = [sample_point_alligned_parallelogram_2d(min_x, max_x, min_y, max_y) for _ in range(n_samples)]
    return samples

def sample_point_parallelogram_3d(normal1: Tuple[float, float, float], normal2: Tuple[float, float, float], 
                                  normal3: Tuple[float, float, float], center: Tuple[float, float, float], 
                                  length1: float, length2: float, length3: float) -> Tuple[float, float, float]:
    '''
    Sample a point from a parallelogram with sides parallel to the vectors normal1, normal2 and normal3.

    :param normal1: Tuple[float, float, float]
        The first vector normal to the sides of the parallelogram.
    :param normal2: Tuple[float, float, float]
        The second vector normal to the sides of the parallelogram.
    :param normal3: Tuple[float, float, float]
        The third vector normal to the sides of the parallelogram.
    :param center: Tuple[float, float, float]
        The center of the parallelogram.
    :param length1: float
        The length of the first side of the parallelogram.
    :param length2: float
        The length of the second side of the parallelogram.    
    :param length3: float
        The length of the third side of the parallelogram.
    :return: Tuple[float, float, float]
        A tuple (x, y, z) representing the coordinates of the sampled point.
    '''
    # Generate two random numbers between -length1/2 and length1/2 and -length2/2 and length2/2 respectively
    x = random.uniform(-length1/2, length1/2)
    y = random.uniform(-length2/2, length2/2)
    z = random.uniform(-length3/2, length3/2)
    # Those numbers represent the coordinates of the sampled point in the normal coordinate system
    # Transform the coordinates to the global coordinate system
    projected_point = x * np.array(normal1) + y * np.array(normal2) + z * np.array(normal3) + np.array(center)
    return projected_point

def sampling_parallelogram_3d(n_samples: int, normal1: Tuple[float, float, float], normal2: Tuple[float, float, float],
                              normal3: Tuple[float, float, float], center: Tuple[float, float, float], 
                              length1: float, length2: float, length3: float) -> List[Tuple[float, float, float]]:
    normal1 = np.array(normal1)
    normal2 = np.array(normal2)
    normal3 = np.array(normal3)
    center = np.array(center)
    samples = [sample_point_parallelogram_3d(normal1, normal2, normal3, center, length1, length2, length3) for _ in range(n_samples)]
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

def sample_point_sphere(center, radius=1):
    """
    Generate a random point on a sphere with a specified radius and center, using rejection sampling.

    :param center: tuple
        A tuple (x, y, z) representing the center of the circle.
    :param radius: float, optional
        The radius of the circle. Default is 1.
    :return: tuple
        A tuple (x, y, z) representing the coordinates of the sampled point.
    """
    while True:
        # Generate a random point on the circle with radius 1 centered at (0,0)
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        z = random.uniform(-1, 1)
        
        # Scale the coordinates to the desired radius
        x *= radius
        y *= radius
        z *= radius
        
        # Translate the point to the desired center
        x += center[0]
        y += center[1]
        z += center[2]
        
        # Check if the point is within the circle
        if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2:
            return x, y, z

def sampling_sphere(n_samples=1, center=(0,0,0), radius=1, seed=None):
    """
    Generate random samples on a sphere with a specified radius and center.

    :param n_samples: int
        The number of random samples to generate on the circle.
    :param center: tuple, optional
        A tuple (x, y, z) representing the center of the circle. Default is (0,0).
    :param radius: float, optional
        The radius of the circle. Default is 1.
    :param seed: int, optional
        The seed value for the random number generator. Default is None.
    :return: list of tuple
        A list of tuples (x, y, z) representing the coordinates of the sampled points.
        
    .. note:: The samples are uniformly distributed on the sphere.
    
    Examples
    --------
    Generate 100 random samples on a sphere with center (2,3,1) and radius 5:
    
    >>> samples = sampling_circle_2d(100, center=(2,3,1), radius=5)
    
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
    """
    # Set the random seed for reproducibility if provided
    if seed is not None:
        set_random_seed(seed)

    # Generate the specified number of samples on the circle
    samples = [sample_point_sphere(center, radius) for _ in range(n_samples)]

    return samples

def sampling_np_array_points(points, num_points = 1, len_points=None):
    if len_points is None:
        len_points = len(points)
    random_points_indices = np.random.choice(range(len_points), num_points, replace=False)
    random_points = points[random_points_indices]
    return random_points

def sampling_pcd_points(pcd, num_points = 1):
    points = np.asarray(pcd.points)
    random_points = sampling_np_array_points(points, num_points)
    return random_points
