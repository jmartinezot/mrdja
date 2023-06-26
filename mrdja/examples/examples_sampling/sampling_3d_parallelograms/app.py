'''
NOT WORKING!!!! ABANDONED TILL I FIND A GOOD WAY OF SHOWING 3D INTERACTIVE PLOTS IN GRADIO
'''


import gradio as gr
import mrdja.sampling as sampling
import mrdja.geometry as geometry
import matplotlib.pyplot as plt
from gradio.components import Number, Image
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')

def plot_parallelogram(n_samples, normal1_x, normal1_y, normal1_z, normal2_x, normal2_y, normal2_z,
                       normal3_x, normal3_y, normal3_z, center_x, center_y, center_z, length1, length2, length3):
    samples = sampling.sampling_parallelogram_3d(n_samples=n_samples, normal1=(normal1_x, normal1_y, normal1_z), 
                                                 normal2=(normal2_x, normal2_y, normal2_z), 
                                                 normal3=(normal3_x, normal3_y, normal3_z),
                                                 center=(center_x, center_y, center_z), 
                                                 length1=length1, length2=length2, length3=length3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])

    center = np.array([center_x, center_y, center_z])
    normal1 = np.array([normal1_x, normal1_y, normal1_z])
    normal2 = np.array([normal2_x, normal2_y, normal2_z])
    normal3 = np.array([normal3_x, normal3_y, normal3_z])
    vertices = geometry.get_parallelogram_3d_vertices(center, normal1, normal2, normal3, length1, length2, length3)
    xlim_min = np.min(vertices[:, 0])
    xlim_max = np.max(vertices[:, 0])
    ylim_min = np.min(vertices[:, 1])
    ylim_max = np.max(vertices[:, 1])
    zlim_min = np.min(vertices[:, 2])
    zlim_max = np.max(vertices[:, 2])
    graph_limits = geometry.get_limits_of_3d_graph_from_limits_of_object(xlim_min, xlim_max, ylim_min, ylim_max, zlim_min, zlim_max)
    ax.set_xlim(graph_limits[0], graph_limits[1])  # Set x-axis limits
    ax.set_ylim(graph_limits[2], graph_limits[3])  # Set y-axis limits
    ax.set_zlim(graph_limits[4], graph_limits[5])  # Set z-axis limits
    # create title from n_samples, center, and radius, using f-string
    ax.set_title(f'{n_samples} Samples on a Parallelogram with normal vectors {normal1} and {normal2}, center {center}, length1 {length1}, and length2 {length2}')
    # draw also the parallelogram in red
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
            (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    # Plot the edges
    for edge in edges:
        ax.plot([vertices[edge[0]][0], vertices[edge[1]][0]],
                [vertices[edge[0]][1], vertices[edge[1]][1]],
                [vertices[edge[0]][2], vertices[edge[1]][2]], color='red')

    # Draw the X, Y and Z axes as dotted lines
    ax.plot([0, 0], [0, 0], [-graph_limits[5], graph_limits[5]], 'k--')
    ax.plot([0, 0], [-graph_limits[3], graph_limits[3]], [0, 0], 'k--')
    ax.plot([-graph_limits[1], graph_limits[1]], [0, 0], [0, 0], 'k--')

    # Draw the normals at a quarter of their corresponding length
    quarter_length1 = length1 / 8
    quarter_length2 = length2 / 8
    quarter_length3 = length3 / 8
    arrow_length1 = quarter_length1 / 2
    arrow_length2 = quarter_length2 / 2
    arrow_length3 = quarter_length3 / 2
    ax.arrow(center_x, center_y, center_z, 
                normal1_x * quarter_length1, normal1_y * quarter_length1, normal1_z * quarter_length1, 
                head_width=arrow_length1, head_length=arrow_length1, fc='b', ec='b')
    ax.arrow(center_x, center_y, center_z,
                normal2_x * quarter_length2, normal2_y * quarter_length2, normal2_z * quarter_length2,
                head_width=arrow_length2, head_length=arrow_length1, fc='b', ec='b')
    ax.arrow(center_x, center_y, center_z,
                normal3_x * quarter_length3, normal3_y * quarter_length3, normal3_z * quarter_length3,  
                head_width=arrow_length3, head_length=arrow_length1, fc='b', ec='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

# Define the input and output interfaces for Gradio
inputs = [
    Number(value=100, precision=0, label="Number of Samples"),
    Number(value=1, label="X value of the first normal vector"),
    Number(value=0, label="Y value of the first normal vector"),
    Number(value=0, label="Z value of the first normal vector"),
    Number(value=0, label="X value of the second normal vector"),
    Number(value=1, label="Y value of the second normal vector"),
    Number(value=0, label="Z value of the second normal vector"),
    Number(value=0, label="X value of the third normal vector"),
    Number(value=0, label="Y value of the third normal vector"),
    Number(value=1, label="Z value of the third normal vector"),
    Number(value=0, label="X value of the center of the parallelogram"),
    Number(value=0, label="Y value of the center of the parallelogram"),
    Number(value=1, label="Z value of the center of the parallelogram"),
    Number(value=1, label="Length of the first side of the parallelogram"),
    Number(value=1, label="Length of the second side of the parallelogram"),
    Number(value=1, label="Length of the third side of the parallelogram")
]
outputs = [
    Image(label="Parallelogram Plot", type="pil")
]

# Create the Gradio app
app = gr.Interface(
    fn=plot_parallelogram, 
    inputs=inputs, 
    outputs=outputs,
    title="Sampling 3D Parallelograms",
    description="Generate random samples on a 3D parallelogram",
    allow_flagging="never",
)

# Run the app
app.launch()
