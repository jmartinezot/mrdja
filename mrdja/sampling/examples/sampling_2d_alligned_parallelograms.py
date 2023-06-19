import mrdja.sampling as sampling
import matplotlib.pyplot as plt

n_samples = 100
min_x = -3
max_x = 2
min_y = -1
max_y = 5
samples = sampling.sampling_alligned_parallelogram_2d(n_samples, min_x, max_x, min_y, max_y)
fig, ax = plt.subplots()
ax.scatter(*zip(*samples))
ax.set_aspect('equal')
# create title from n_samples, center, and radius, usign fstring
ax.set_title(f'{n_samples} Samples on a axes alligned Parallelogram with bottom left corner {min_x}, {min_y} and top right corner {min_y}, {max_y}')
# draw also the parallelogram in red
ax.plot([min_x, max_x], [min_y, min_y], color='r')
ax.plot([min_x, max_x], [max_y, max_y], color='r')
ax.plot([min_x, min_x], [min_y, max_y], color='r')
ax.plot([max_x, max_x], [min_y, max_y], color='r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()
