import gradio as gr
import mrdja.sampling as sampling
import mrdja.geometry as geometry
import matplotlib.pyplot as plt
from gradio.components import Number, Image
import matplotlib
import numpy as np
matplotlib.use('Agg')

def plot_alligned_parallelogram(n_samples, min_x, max_x, min_y, max_y):
    samples = sampling.sampling_alligned_parallelogram_2d(n_samples=n_samples, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
    fig, ax = plt.subplots()
    ax.scatter(*zip(*samples))
    ax.set_aspect('equal')
    xlim_min = min_x
    xlim_max = max_x
    ylim_min = min_y
    ylim_max = max_y
    graph_limits = geometry.get_limits_of_graph_from_limits_of_object(xlim_min, xlim_max, ylim_min, ylim_max)
    ax.set_xlim(graph_limits[0], graph_limits[1])  # Set x-axis limits
    ax.set_ylim(graph_limits[2], graph_limits[3])  # Set y-axis limits
    # create title from n_samples, center, and radius, using f-string
    ax.set_title(f'{n_samples} Samples on a axes alligned Parallelogram with bottom left corner {min_x}, {min_y} and top right corner {max_x}, {max_y}')
    # draw also the parallelogram in red
    ax.plot([min_x, max_x], [min_y, min_y], color='r')
    ax.plot([min_x, max_x], [max_y, max_y], color='r')
    ax.plot([min_x, min_x], [min_y, max_y], color='r')
    ax.plot([max_x, max_x], [min_y, max_y], color='r')

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
    Number(value=0, label="X value of bottom left corner"),
    Number(value=0, label="X value of top right corner"),
    Number(value=1, label="Y value of bottom left corner"),
    Number(value=1, label="Y value of top right corner")
]
outputs = [
    Image(label="Parallelogram Plot", type="pil")
]

# Create the Gradio app
app = gr.Interface(
    fn=plot_alligned_parallelogram, 
    inputs=inputs, 
    outputs=outputs,
    title="Sampling 2D Alligned Parallelograms",
    description="Generate random samples on a 2D alligned parallelogram",
    allow_flagging="never",
)

# Run the app
app.launch()
