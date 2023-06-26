import mrdja.sampling as sampling
import matplotlib.pyplot as plt

n_samples = 100
center = (2,3)
radius = 5
samples = sampling.sampling_circle_2d(n_samples=n_samples, center=center, radius=radius)
fig, ax = plt.subplots()
ax.scatter(*zip(*samples))
ax.set_aspect('equal')
# create title from n_samples, center, and radius, usign fstring
ax.set_title(f'{n_samples} Samples on a Circle with Center {center} and Radius {radius}')
# draw also the circle in red
circle = plt.Circle(center, radius, color='r', fill=False)
ax.add_artist(circle)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()