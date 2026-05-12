import os
import sys
sys.path.append(os.getcwd())

import trimesh
import plotly.graph_objects as go
import numpy as np

def as_trimesh(mesh_or_scene):
    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = [
            geom for geom in mesh_or_scene.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]
        if len(meshes) == 0:
            raise ValueError("Scene 안에 Trimesh geometry가 없습니다.")
        return trimesh.util.concatenate(meshes)
    return mesh_or_scene


def center_mesh(mesh):
    mesh = mesh.copy()
    center = mesh.bounds.mean(axis=0)  # bbox center
    mesh.apply_translation(-center)
    return mesh


def add_trimesh_to_fig(fig, mesh, name="mesh", color="blue", opacity=1.0):
    vertices = mesh.vertices
    faces = mesh.faces

    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=name,
            color=color,
            opacity=opacity,
            flatshading=True,
            showscale=False
        )
    )

def save_multiple_meshes_grid_html(
    mesh_paths,
    save_path="multiple_meshes_grid.html",
    distance=2.0,
    items_per_row=5,
    diagonal_shift=0.5
):
    fig = go.Figure()

    colors = [
        "blue", "red", "green", "orange", "purple",
        "cyan", "magenta", "brown", "pink", "gray"
    ]

    n = len(mesh_paths)
    if n == 0:
        raise ValueError("mesh_paths가 비어 있습니다.")

    # xy 평면 기준 반시계 방향 45도 회전 = z축 기준 +45도 회전
    angle = np.deg2rad(45)
    R = trimesh.transformations.rotation_matrix(
        angle=angle,
        direction=[0, 0, 1],
        point=[0, 0, 0]
    )

    for idx, mesh_path in enumerate(mesh_paths):
        mesh = trimesh.load(mesh_path, process=False)
        mesh = as_trimesh(mesh)
        mesh = center_mesh(mesh)

        # object 자체를 회전
        mesh.apply_transform(R)

        row = idx // items_per_row
        col = idx % items_per_row

        x = col * distance
        y = -row * distance - col * diagonal_shift
        z = 0.0

        mesh.apply_translation([x, y, z])

        name = os.path.basename(os.path.dirname(os.path.dirname(mesh_path)))
        color = colors[idx % len(colors)]

        add_trimesh_to_fig(
            fig,
            mesh,
            name=name,
            color=color,
            opacity=1.0
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
    print(f"saved to {save_path}")


def main():

    ROOT = "DexGraspNet/data/meshdata"
    SAVE_DIR = "vis"
    os.makedirs(SAVE_DIR, exist_ok=True)

    prefix = "core-bottle"
    mesh_paths = [f"{ROOT}/{fname}/coacd/decomposed.obj" for fname in os.listdir(ROOT) if fname.startswith(prefix)]

    save_multiple_meshes_grid_html(
        mesh_paths,
        save_path=f"{SAVE_DIR}/all_objects_side_by_side.html",
        distance=2.0,
        items_per_row=10
    )

if __name__ == "__main__":
    main()