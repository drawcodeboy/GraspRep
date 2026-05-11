import os, sys
import numpy as np
import pytorch_kinematics as pk
import trimesh as tm
import torch

GRASP_BASE_PATH = f"DexGraspNet/data/dexgraspnet"
MJCF_PATH = f"DexGraspNet/grasp_generation/mjcf/shadow_hand_vis.xml"
MESH, AREAS = {}, {}

def get_robot_model(device):
    # Robot model의 mesh를 가져와야 하는데, 그게 현재 경로 기반으로 되어있어서
    # cwd 이동으로 해당 경로로 가서 가져와주고 robot model을 로드했을 때, 원래 경로로 돌아오게 작업

    mjcf_related_path = os.path.dirname(os.path.dirname(MJCF_PATH))
    old_cwd = os.getcwd()

    os.chdir(mjcf_related_path) 
    chain = pk.build_chain_from_mjcf(open(os.path.join(os.getcwd(), "mjcf/shadow_hand_vis.xml")).read()).to(dtype=torch.float, device=device)
    os.chdir(old_cwd)

    return chain

def index_vertices_by_faces(vertices_features, faces):
    r"""Index vertex features to convert per vertex tensor to per vertex per face tensor.

    Args:
        vertices_features (torch.FloatTensor):
            vertices features, of shape
            :math:`(\text{batch_size}, \text{num_points}, \text{knum})`,
            ``knum`` is feature dimension, the features could be xyz position,
            rgb color, or even neural network features.
        faces (torch.LongTensor):
            face index, of shape :math:`(\text{num_faces}, \text{num_vertices})`.
    Returns:
        (torch.FloatTensor):
            the face features, of shape
            :math:`(\text{batch_size}, \text{num_faces}, \text{num_vertices}, \text{knum})`.
    """
    assert vertices_features.ndim == 2, \
        "vertices_features must have 2 dimensions of shape (batch_sizenum_points, knum)"
    assert faces.ndim == 2, "faces must have 2 dimensions of shape (num_faces, num_vertices)"
    # input = vertices_features.unsqueeze(2).expand(-1, -1, faces.shape[-1], -1)
    # indices = faces[None, ..., None].expand(
    #     vertices_features.shape[0], -1, -1, vertices_features.shape[-1])
    # return torch.gather(input=input, index=indices, dim=1)
    input = vertices_features.reshape(-1, 1, 3).expand(-1, faces.shape[-1], -1)
    indices = faces[..., None].expand(
        -1, -1, vertices_features.shape[-1])
    return torch.gather(input=input, index=indices, dim=0)

def build_mesh_recurse(chain_root, mesh_path, device):
    # chain_root(chain._root)는 로봇 kinematic tree의 root body/node이다.
    # chain_root.link는 그 body에 붙어 있는 link 정보이다.

    if len(chain_root.link.visuals) > 0:
        link_name = chain_root.link.name
        link_vertices = []
        link_faces = []
        n_link_vertices = 0

        for visual in chain_root.link.visuals:
            print(visual)
            # Hand mesh를 구성하는 각 visual element에 대해서, mesh를 가져와서 vertex와 face 정보를 추출한다.
            scale = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
            if visual.geom_type == 'box':
                link_mesh = tm.load_mesh(os.path.join(mesh_path, 'box.obj'), process=False) # 원본 mesh 그대로 들고 올 수 있도록
                link_mesh.vertices *= visual.geo_param.detach().cpu().numpy() # mesh vertex 좌표들에 곱할 scale
            elif visual.geom_type == "capsule":
                link_mesh = tm.primitives.Capsule(radius=visual.geom_param[0], height=visual.geom_param[1] * 2).apply_translation((0, 0, -visual.geom_param[1]))
            # elif visual.geom_type == "mesh":
            else:
                link_mesh = tm.load_mesh(os.path.join(mesh_path, visual.geom_param[0].split(":")[1]+".obj"), process=False)
                if visual.geom_param[1] is not None:
                    scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=device)
            vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
            pos = visual.offset.to(device)
            vertices = vertices * scale
            vertices = pos.transform_points(vertices)
            link_vertices.append(vertices)
            link_faces.append(faces + n_link_vertices)
            n_link_vertices += len(vertices)

        link_vertices = torch.cat(link_vertices, dim=0)
        link_faces = torch.cat(link_faces, dim=0)

        MESH[link_name] = {
            'vertices': link_vertices,
            'faces': link_faces
        }

        if link_name in ['robot0:palm', 'robot0:palm_child', 'robot0:lfmetacarpal_child']:
            # 여기 TorchSDF 라이브러리에서 가져옴
            link_face_verts = index_vertices_by_faces(link_vertices, link_faces)
            MESH[link_name]['face_verts'] = link_face_verts
        else:
            MESH[link_name]['geom_param'] = chain_root.link.visuals[0].geom_param
        AREAS[link_name] = tm.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()

def process_param(grasp_param):
    pass

def main():
    device = 'cuda' if  torch.cuda.is_available() else 'cpu'
    
    # Robot model 가져오기
    chain = get_robot_model(device)
    # print(f"n_dofs:{len(chain.get_joint_parameter_names())}")

    mesh_path = 'DexGraspNet/grasp_generation/mjcf/meshes'
    build_mesh_recurse(chain._root, mesh_path, device)

    for idx, fname in enumerate(os.listdir(GRASP_BASE_PATH), start=1):
        if idx == 2: break

        file_path = os.path.join(GRASP_BASE_PATH, fname)
        grasp_params = np.load(file_path, allow_pickle=True)

        for grasp_param in grasp_params:
            _ = process_param(grasp_param)

if __name__ == '__main__':
    main()