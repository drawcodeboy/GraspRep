#!/usr/bin/env python3
"""Sample DexGraspNet mesh decompositions and save point clouds to .npy.

This script scans `DexGraspNet/data/meshdata`, finds object folders that
contain `coacd/decomposed.obj`, samples surface points, and saves them as a
NumPy array. The default behavior is to write a .npy file, so no explicit save
flag is required.

Example:
    python subtasks/02_visualize_meshdata.py --object-code sem-Ball-29830fb806fe23b29ccf01d06bf2094d --sample-points 2000
    python subtasks/02_visualize_meshdata.py --prefix sem-Ball --random --sample-points 4096 --output sampled.npy
"""

import argparse
import os
import random
import sys

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

DEFAULT_ROOT = os.path.join("DexGraspNet", "data", "meshdata")
DEFAULT_SAMPLE_POINTS = 2048


def find_mesh_objects(root_path):
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"Mesh root path not found: {root_path}")

    for object_code in sorted(os.listdir(root_path)):
        object_dir = os.path.join(root_path, object_code)
        if not os.path.isdir(object_dir):
            continue

        decomposed_path = os.path.join(object_dir, "coacd", "decomposed.obj")
        if os.path.isfile(decomposed_path):
            yield object_code, decomposed_path



def load_mesh(obj_path):
    if trimesh is None:
        raise ImportError("trimesh is required to load mesh files. Install it with `pip install trimesh`.")
    mesh = trimesh.load(obj_path, force="scene", process=False)
    if isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Scene(mesh)
    return mesh



def scene_to_mesh(mesh):
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError("Scene contains no geometry")
        return trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh



def sample_surface_points(mesh, count):
    mesh = scene_to_mesh(mesh)
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    return mesh.sample(count)



def save_point_cloud_npy(points, out_path):
    if not out_path.lower().endswith(".npy"):
        raise ValueError("Output path must end with .npy")
    np.save(out_path, points)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample DexGraspNet decomposed.obj meshes and save surface point clouds to .npy"
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Path to DexGraspNet/data/meshdata")
    parser.add_argument("--object-code", help="Exact object folder name to process")
    parser.add_argument("--prefix", help="Only consider object codes starting with this prefix")
    parser.add_argument("--random", action="store_true", help="Choose a random object from the available set")
    parser.add_argument("--sample-points", type=int, default=DEFAULT_SAMPLE_POINTS, help="Number of surface points to sample")
    parser.add_argument("--output", help="Output .npy file path. Defaults to <object_code>-<sample_points>.npy")
    return parser.parse_args()



def main():
    args = parse_args()
    object_entries = list(find_mesh_objects(args.root))

    if args.prefix:
        object_entries = [entry for entry in object_entries if entry[0].startswith(args.prefix)]

    if not object_entries:
        print(f"No decomposed.obj objects found in {args.root}" + (f" matching prefix {args.prefix!r}" if args.prefix else ""))
        sys.exit(1)

    if args.object_code:
        object_entries = [entry for entry in object_entries if entry[0] == args.object_code]
        if not object_entries:
            print(f"Object code not found: {args.object_code}")
            sys.exit(1)

    if args.random:
        object_code, obj_path = random.choice(object_entries)
    else:
        object_code, obj_path = object_entries[0]

    output_path = args.output
    if output_path is None:
        # output_path = f"{object_code}-{args.sample_points}.npy"
        output_path = "output.npy"

    print(f"Selected object: {object_code}")
    print(f"Mesh path: {obj_path}")
    print(f"Sampling {args.sample_points} points and saving to {output_path}")

    mesh = load_mesh(obj_path)
    points = sample_surface_points(mesh, args.sample_points)
    save_point_cloud_npy(points, output_path)

    print(f"Saved {points.shape[0]} points to {output_path}")


if __name__ == "__main__":
    main()
