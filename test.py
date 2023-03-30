import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from gradio.components import Plot
import plotly.graph_objects as go

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

def plot_3d_circle_samples(radius, n_samples, x, y, z, nx, ny, nz):
    center = np.array([x, y, z])
    normal = np.array([nx, ny, nz])

    samples = sampling_circle_3d_rejection(n_samples, radius, center, normal)
    
    xs, ys, zs = zip(*samples)
    fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker=dict(size=3))])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

    return fig

inputs = [
    gr.inputs.Slider(0.1, 5, step=0.1, default=1, label='Radius'),
    gr.inputs.Slider(10, 1000, step=10, default=100, label='Number of Samples'),
    gr.inputs.Number(default=0, label='Center X'),
    gr.inputs.Number(default=0, label='Center Y'),
    gr.inputs.Number(default=0, label='Center Z'),
    gr.inputs.Number(default=0, label='Normal X'),
    gr.inputs.Number(default=0, label='Normal Y'),
    gr.inputs.Number(default=1, label='Normal Z'),
]

iface = gr.Interface(
    fn=plot_3d_circle_samples,
    inputs=inputs,
    outputs=Plot(label="points"),
    title="3D Circle Sampling using Rejection Sampling",
    description="Visualize random points on a 3D circle generated using rejection sampling."
)

iface.launch()
