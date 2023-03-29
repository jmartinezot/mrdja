import random
import math

def sample_point(radius=1):
    while True:
        x = random.uniform(-radius, radius)
        y = random.uniform(-radius, radius)
        if x**2 + y**2 <= radius**2:
            return x, y

def rejection_sampling_circle(n_samples, radius=1):
    samples = [sample_point(radius) for _ in range(n_samples)]
    return samples

