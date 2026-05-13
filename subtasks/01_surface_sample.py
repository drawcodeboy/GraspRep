import os, sys
import numpy as np
import pytorch_kinematics as pk
import trimesh as tm
import torch
import transforms3d
import plotly.graph_objects as go
import time

DATA_BASE_PATH = f"DexGraspNet/data"
MJCF_PATH = f"DexGraspNet/grasp_generation/mjcf/shadow_hand_vis.xml"
MESH, AREAS = {}, {}
GLOBAL_TRANSLATION, GLOBAL_ROTATION, CURRENT_STATUS = None, None, None
N_POINTS = 2048
MUL_POINTS = 8

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
    # TorchSDF에서 Copy & paste
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
            # Hand mesh를 구성하는 각 visual element에 대해서, mesh를 가져와서 vertex와 face 정보를 추출한다.
            scale = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
            if visual.geom_type == 'box':
                link_mesh = tm.load_mesh(os.path.join(mesh_path, 'box.obj'), process=False) # 원본 mesh 그대로 들고 올 수 있도록
                link_mesh.vertices *= visual.geom_param.detach().cpu().numpy() # mesh vertex 좌표들에 곱할 scale
            elif visual.geom_type == "capsule":
                link_mesh = tm.primitives.Capsule(radius=visual.geom_param[0], height=visual.geom_param[1] * 2).apply_translation((0, 0, -visual.geom_param[1]))
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
        # Mesh의 전체 삼각형 face 면적을 합산한 총 표면적 계산
        AREAS[link_name] = tm.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()
    for children in chain_root.children:
        build_mesh_recurse(children, mesh_path, device)

def farthest_point_sampling(points, K):
    """
    points: (N, 3) torch.Tensor
    return: (K, 3) torch.Tensor
    """
    N = points.shape[0]

    if K <= 0:
        return points.new_empty((0, 3))

    if N <= K:
        return points

    selected_idx = torch.empty(K, dtype=torch.long, device=points.device)

    # 첫 point는 random하게 선택
    farthest = torch.randint(0, N, (1,), device=points.device).item()

    # 각 point가 selected set까지 가지는 최소 거리
    min_dist = torch.full((N,), float("inf"), device=points.device)

    for i in range(K):
        selected_idx[i] = farthest

        selected_point = points[farthest].unsqueeze(0)  # (1, 3)
        dist = torch.sum((points - selected_point) ** 2, dim=1)  # (N,)

        min_dist = torch.minimum(min_dist, dist)
        farthest = torch.argmax(min_dist).item()

    return points[selected_idx]

def sample_surface_points(n_surface_points, device):
    total_area = sum(AREAS.values())
    # 각 link의 표면적 비율에 따라 n_surface_points 개수만큼 surface point를 sampling하기 위한 개수 계산
    num_samples = dict([(link_name, int(AREAS[link_name] / total_area * n_surface_points)) for link_name in MESH])
    # num_samples의 총합이 n_surface_points보다 작을 수 있기 때문에, 가장 면적이 큰 link에 부족한 개수 추가
    # num_samples의 첫 번째 인자가 가장 큰 면적을 갖는 link라는 것이 상정되는 이유는 forward kinematics로 인해 저기가 palm이 될 것이라 그런 듯
    num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(num_samples.values()) 
    
    for link_name in MESH:
        if num_samples[link_name] == 0:
            # empty tensor지만, shape이 (0, 3)으로 잡히긴 함.
            MESH[link_name]['surface_points'] = torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
            continue
        # w/o pytorch3d
        vertices_np = MESH[link_name]['vertices'].detach().cpu().numpy()
        faces_np = MESH[link_name]['faces'].detach().cpu().numpy()
        tri_mesh = tm.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
        dense_points_np, _ = tm.sample.sample_surface(
                                tri_mesh,
                                count=MUL_POINTS * num_samples[link_name]
                            )
        dense_point_cloud = torch.tensor(
            dense_points_np,
            dtype=torch.float32,
            device=device
        )

        # 기존 코드의 sample_farthest_points 역할
        surface_points = farthest_point_sampling(
            dense_point_cloud,
            K=num_samples[link_name]
        )
        MESH[link_name]['surface_points'] = surface_points

def qpos_to_handpose(qpos):
    joint_names = [
        'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    ]
    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

    rot = np.array(transforms3d.euler.euler2mat(
        *[qpos[name] for name in rot_names]
    ))

    rot = rot[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name]
                            for name in joint_names], dtype=torch.float, device="cpu").unsqueeze(0)
    
    return hand_pose

def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        
    return out

def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix

def set_parameters(chain, qpos, device):
    hand_pose = qpos_to_handpose(qpos)
    hand_pose = torch.tensor(hand_pose, dtype=torch.float, device=device)
    
    global GLOBAL_TRANSLATION, GLOBAL_ROTATION, CURRENT_STATUS
    GLOBAL_TRANSLATION = hand_pose[:, 0:3]
    GLOBAL_ROTATION = robust_compute_rotation_matrix_from_ortho6d(hand_pose[:, 3:9])
    CURRENT_STATUS = chain.forward_kinematics(hand_pose[:, 9:])

def get_surface_points(device):
    # For Hand mesh
    points = [] 
    batch_size = GLOBAL_ROTATION.shape[0]

    for link_name in MESH:
        ## 여기부터 작업하면 됨
        n_surface_points = MESH[link_name]['surface_points'].shape[0]
        
        # NOTE: point cloud를 forward kinematics, angle parameter에 따라 rigid transformation
        points.append(CURRENT_STATUS[link_name].transform_points(MESH[link_name]['surface_points']))
    
    points = torch.cat(points, dim=-2).to(device)
    points = points @ GLOBAL_ROTATION.transpose(1, 2) + GLOBAL_TRANSLATION.unsqueeze(1)
    return points

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
    device = 'cuda' if  torch.cuda.is_available() else 'cpu'
    
    # Robot model 가져오기
    chain = get_robot_model(device)
    # print(f"n_dofs:{len(chain.get_joint_parameter_names())}")

    mesh_path = 'DexGraspNet/grasp_generation/mjcf/meshes'
    build_mesh_recurse(chain._root, mesh_path, device)

    # Set joint limits (이 부분은 pass) / 움직임 범위 limit 지정은 아직 필요 없어 보임

    # Sample surface points
    sample_surface_points(N_POINTS, device)

    os.makedirs("vis", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    PC_VISUALIZATION_SAVE_BASE_PATH = f"vis/"
    PC_DATA_SAVE_BASE_PATH = f"data/DexGraspNet/"

    os.makedirs(f"{PC_DATA_SAVE_BASE_PATH}/pointcloud", exist_ok=True)
    
    grasp_data_dir = os.listdir(f"{DATA_BASE_PATH}/dexgraspnet")

    total_start_time = time.time()
    for file_idx, fname in enumerate(grasp_data_dir, start=1):
        obj_start_time = time.time()

        hand_pose_file_path = os.path.join(f"{DATA_BASE_PATH}/dexgraspnet", fname)
        hand_poses = np.load(hand_pose_file_path, allow_pickle=True)

        object_file_path = os.path.join(f"{DATA_BASE_PATH}/meshdata", fname.split(".")[0])
        object_file_path = os.path.join(object_file_path, "coacd/decomposed.obj")

        # Object load & FPS (surface sampling)
        object_mesh_origin = tm.load(object_file_path, process=False)
        object_points, face_indices = tm.sample.sample_surface(object_mesh_origin, count=N_POINTS * MUL_POINTS)
        object_points = torch.tensor(object_points, dtype=torch.float32, device=device)
        object_points = farthest_point_sampling(object_points, N_POINTS)

        for handpose_idx, hand_pose in enumerate(hand_poses, start=1):
            qpos, scale = hand_pose['qpos'], hand_pose['scale']
            # NOTE: 이전 chain에 기반해서 FK를 하는 거라면, 위험하지 않을까 모든 grasp이 mean pose(?)로부터 시작해야 하지 않을까
            # 의심스러운 상황일 뿐, 확증은 없는 상태이다.
            set_parameters(chain, qpos, device)
            hand_points = get_surface_points(device)

            # Grasp마다 object의 scale이 다르기 때문에 이를 반영하기 위해 grasp마다 object의 point cloud를 새로 sampling한다.
            scaled_object_points = object_points * float(scale)

            # Naming rule {filename}_{grasp_num(idx):04d}_pc.html
            # save_point_cloud_html(hand_points, scaled_object_points, f"{PC_VISUALIZATION_SAVE_BASE_PATH}/{fname.split('.')[0]}_{handpose_idx:04d}_pc.html")
            
            # Object & Hand, 그리고 scale 정보도 저장하기
            # mesh도 저장
            object_points_np = scaled_object_points.detach().cpu().numpy()
            hand_points_np = hand_points.detach().cpu().numpy()

            sample = {
                "object_id": fname.split(".")[0],
                "grasp_idx": handpose_idx, # start point 1, not 0!
                "hand_pc": hand_points_np,
                "object_pc": object_points_np, # scaling 처리된 pc
                "qpos": qpos, # NumPy 포맷 맞음
                "scale": scale, # NumPy 포맷 맞음
                "object_mesh_path": object_file_path,
                "hand_model_path": MJCF_PATH
            }
            np.save(f"{PC_DATA_SAVE_BASE_PATH}/pointcloud/{fname.split('.')[0]}-{handpose_idx:05d}.npy", sample)

            if handpose_idx % 20 == 0:
                print(f"[Obj&Grasps file {file_idx:04d}/{len(grasp_data_dir)}]\t[Grasp index: {handpose_idx:03d}/{len(hand_poses):03d}]")
        obj_elapsed_time = time.time() - obj_start_time
        print(f"File processing elapsed time: {int(obj_elapsed_time//3600):02d}h {int(obj_elapsed_time%3600)//60:02d}m {int(obj_elapsed_time)%60:02d}s")
    
    total_elapsed_time = time.time() - total_start_time
    print(f"Total elapsed time: {int(total_elapsed_time//3600):02d}h {int(total_elapsed_time%3600)//60:02d}m {int(total_elapsed_time)%60:02d}s")

if __name__ == '__main__':
    main()