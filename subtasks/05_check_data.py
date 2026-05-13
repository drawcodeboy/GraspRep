import os, sys
sys.path.append(os.getcwd())
import numpy as np
import plotly.graph_objects as go
import torch

def save_point_cloud_html(points1, points2, save_path,
                          name1="point_cloud_1", name2="point_cloud_2",
                          color1="blue", color2="red",
                          size1=2, size2=2):
    """
    points1: torch.Tensor, shape (1, N, 3) or (N, 3)
    points2: torch.Tensor, shape (1, M, 3) or (M, 3)
    """

    if points1.dim() == 3:
        points1 = points1[0]
    if points2.dim() == 3:
        points2 = points2[0]

    pts1 = points1.detach().cpu().numpy()
    pts2 = points2.detach().cpu().numpy()

    opacity = 1.0

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts1[:, 0],
                y=pts1[:, 1],
                z=pts1[:, 2],
                mode="markers",
                name=name1,
                marker=dict(
                    size=size1,
                    color=color1,
                    opacity=opacity
                )
            ),
            go.Scatter3d(
                x=pts2[:, 0],
                y=pts2[:, 1],
                z=pts2[:, 2],
                mode="markers",
                name=name2,
                marker=dict(
                    size=size2,
                    color=color2,
                    opacity=opacity
                )
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig.write_html(save_path)

def main():
    PROCESSED_DATA_PATH = "data/DexGraspNet/pointcloud"
    data_dir = os.listdir(PROCESSED_DATA_PATH)
    data = np.load(os.path.join(PROCESSED_DATA_PATH, data_dir[100]), allow_pickle=True)

    data = data.item()
    print(type(data))
    save_point_cloud_html(torch.tensor(data['hand_pc'], dtype=torch.float),
                          torch.tensor(data['object_pc'], dtype=torch.float),
                          save_path="vis/example.html")

if __name__ == '__main__':
    main()