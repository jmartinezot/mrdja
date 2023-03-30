import gradio as gr
import numpy as np
import plotly.graph_objects as go
from gradio.components import Plot
import random

def sample_point_parallelogram_3d(a, b, c, u, v):
    return a + u * b + v * c

def sampling_parallelogram_3d(n_samples, a, b, c):
    samples = [sample_point_parallelogram_3d(a, b, c, random.random(), random.random()) for _ in range(n_samples)]
    return samples

def plot_shape_parallelogram(n_samples, x, y, z, bx, by, bz, cx, cy, cz):
    center = np.array([x, y, z])
    b = np.array([bx, by, bz])
    c = np.array([cx, cy, cz])

    samples = sampling_parallelogram_3d(n_samples, center, b, c)

    fig = go.Figure(data=[go.Scatter3d(x=[p[0] for p in samples],
                                       y=[p[1] for p in samples],
                                       z=[p[2] for p in samples],
                                       mode='markers',
                                       marker=dict(size=3))])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    return fig

iface_parallelogram = gr.Interface(
    fn=plot_shape_parallelogram,
    inputs=[
        gr.inputs.Slider(10, 1000, step=10, default=100, label='Number of Samples'),
        gr.inputs.Number(default=0, label='Center X'),
        gr.inputs.Number(default=0, label='Center Y'),
        gr.inputs.Number(default=0, label='Center Z'),
        gr.inputs.Number(default=1, label='B Vector X'),
        gr.inputs.Number(default=0, label='B Vector Y'),
        gr.inputs.Number(default=0, label='B Vector Z'),
        gr.inputs.Number(default=0, label='C Vector X'),
        gr.inputs.Number(default=1, label='C Vector Y'),
        gr.inputs.Number(default=0, label='C Vector Z'),
    ],
    outputs=Plot(label="points"),
    title="3D Parallelogram Sampling",
    description="Visualize random points on a 3D parallelogram.",
)

iface_parallelogram.launch()