import gradio as gr
import mrdja.sampling as sampling
import mrdja.geometry as geometry
import matplotlib.pyplot as plt
from gradio.components import Number, Image
import matplotlib
import numpy as np
matplotlib.use('Agg')

def plot_parallelogram(n_samples, normal1_x, normal1_y, normal2_x, normal2_y, center_x, center_y, length1, length2):
    samples = sampling.sampling_parallelogram_2d(n_samples=n_samples, normal1=(normal1_x, normal1_y), normal2=(normal2_x, normal2_y), center=(center_x, center_y), length1=length1, length2=length2)
    fig, ax = plt.subplots()
    ax.scatter(*zip(*samples))
    ax.set_aspect('equal')
    center = np.array([center_x, center_y])
    normal1 = np.array([normal1_x, normal1_y])
    normal2 = np.array([normal2_x, normal2_y])
    vertex1 = center + normal1 * length1 / 2 + normal2 * length2 / 2
    vertex2 = center - normal1 * length1 / 2 + normal2 * length2 / 2
    vertex3 = center - normal1 * length1 / 2 - normal2 * length2 / 2
    vertex4 = center + normal1 * length1 / 2 - normal2 * length2 / 2
    xlim_min = min(vertex1[0], vertex2[0], vertex3[0], vertex4[0])
    xlim_max = max(vertex1[0], vertex2[0], vertex3[0], vertex4[0])
    ylim_min = min(vertex1[1], vertex2[1], vertex3[1], vertex4[1])
    ylim_max = max(vertex1[1], vertex2[1], vertex3[1], vertex4[1])
    graph_limits = geometry.get_limits_of_graph_from_limits_of_object(xlim_min, xlim_max, ylim_min, ylim_max)
    ax.set_xlim(graph_limits[0], graph_limits[1])  # Set x-axis limits
    ax.set_ylim(graph_limits[2], graph_limits[3])  # Set y-axis limits
    # create title from n_samples, center, and radius, using f-string
    ax.set_title(f'{n_samples} Samples on a Parallelogram with normal vectors {normal1} and {normal2}, center {center}, length1 {length1}, and length2 {length2}')
    # draw also the parallelogram in red
    vertices = geometry.get_parallelogram_2d_vertices(center, normal1, normal2, length1, length2)
    ax.plot([vertices[0][0], vertices[1][0]], [vertices[0][1], vertices[1][1]], color='r')
    ax.plot([vertices[1][0], vertices[2][0]], [vertices[1][1], vertices[2][1]], color='r')
    ax.plot([vertices[2][0], vertices[3][0]], [vertices[2][1], vertices[3][1]], color='r')
    ax.plot([vertices[3][0], vertices[0][0]], [vertices[3][1], vertices[0][1]], color='r')

    # Draw the X and Y axes in dotted lines
    ax.axhline(0, linestyle='dotted', color='black')
    ax.axvline(0, linestyle='dotted', color='black')

    # Draw the normals at a quarter of their corresponding length
    quarter_length1 = length1 / 8
    quarter_length2 = length2 / 8
    arrow_length1 = quarter_length1 / 2
    arrow_length2 = quarter_length2 / 2
    ax.arrow(center_x, center_y, normal1_x * quarter_length1, normal1_y * quarter_length1, head_width=arrow_length1, head_length=arrow_length2, fc='b', ec='b')
    ax.arrow(center_x, center_y, normal2_x * quarter_length2, normal2_y * quarter_length2, head_width=arrow_length2, head_length=arrow_length1, fc='b', ec='b')

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
    Number(value=0, label="X value of the first normal vector"),
    Number(value=1, label="Y value of the first normal vector"),
    Number(value=1, label="X value of the second normal vector"),
    Number(value=0, label="Y value of the second normal vector"),
    Number(value=0, label="X value of the center of the parallelogram"),
    Number(value=0, label="Y value of the center of the parallelogram"),
    Number(value=1, label="Length of the first side of the parallelogram"),
    Number(value=1, label="Length of the second side of the parallelogram")
]
outputs = [
    Image(label="Parallelogram Plot", type="pil")
]

# Create the Gradio app
app = gr.Interface(
    fn=plot_parallelogram, 
    inputs=inputs, 
    outputs=outputs,
    title="Sampling 2D Parallelograms",
    description="Generate random samples on a 2D parallelogram",
    allow_flagging="never",
)

# Run the app
app.launch()
