import gradio as gr
import mrdja.sampling as sampling
import mrdja.geometry as geometry
import matplotlib.pyplot as plt
from gradio.components import Number, Image
import matplotlib
import numpy as np
matplotlib.use('Agg')

def plot_circle(n_samples, center_x, center_y, radius):
    center = (center_x, center_y)
    samples = sampling.sampling_circle_2d(n_samples=n_samples, center=center, radius=radius)
    fig, ax = plt.subplots()
    ax.scatter(*zip(*samples))
    ax.set_aspect('equal')
    xlim_min = center_x - radius
    xlim_max = center_x + radius
    ylim_min = center_y - radius
    ylim_max = center_y + radius
    graph_limits = geometry.get_limits_of_graph_from_limits_of_object(xlim_min, xlim_max, ylim_min, ylim_max)
    ax.set_xlim(graph_limits[0], graph_limits[1])  # Set x-axis limits
    ax.set_ylim(graph_limits[2], graph_limits[3])  # Set y-axis limits
    # create title from n_samples, center, and radius, using f-string
    ax.set_title(f'{n_samples} Samples on a Circle with Center {center} and Radius {radius}')
    # draw also the circle in red
    circle = plt.Circle(center, radius, color='r', fill=False)
    ax.add_artist(circle)

    # Draw the X and Y axes in dotted lines
    ax.axhline(0, linestyle='dotted', color='black')
    ax.axvline(0, linestyle='dotted', color='black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

# Define the input and output interfaces for Gradio
inputs = [
    Number(value=100, precision=0, label="Number of Samples"),
    Number(value=2, label="Center X-coordinate"),
    Number(value=3, label="Center Y-coordinate"),
    Number(value=5, label="Radius")
]
outputs = [
    Image(label="Circle Plot", type="pil")
]

# Create the Gradio app
app = gr.Interface(
    fn=plot_circle, 
    inputs=inputs, 
    outputs=outputs,
    title="Sampling 2D Circles",
    description="Generate random samples on a 2D circle with a specified radius and center.",
    allow_flagging="never",
)

# Run the app
app.launch()
