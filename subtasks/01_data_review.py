import os
import random
import trimesh
import numpy as np

def main():
    base_path = "DexGraspNet/data"

    mesh_ds = os.path.join(base_path, "meshdata")
    grasp_ds = os.path.join(base_path, "dexgraspnet")
    
    grasp_code = random.choice(os.listdir(mesh_ds))
    print(grasp_code)

    mesh_sample_path = os.path.join(mesh_ds, grasp_code, 'coacd/decomposed.obj')
    grasp_sample_path = os.path.join(grasp_ds, grasp_code) + ".npy"

    obj_mesh = trimesh.load(mesh_sample_path)
    grasp_data = np.load(grasp_sample_path, allow_pickle=True)

    for key in grasp_data[0].keys():
        if key == 'qpos':
            for key_, value_ in grasp_data[0][key].items():
                print(f"\t{key_}:{value_}")
        else:
            print(f"{key}: {grasp_data[0][key]}")

if __name__ == '__main__':
    main()