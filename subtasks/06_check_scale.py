import sys, os
sys.path.append(os.getcwd())

import numpy as np

DATA_BASE_PATH = f"DexGraspNet/data"

def main():
    result_cnt = 0
    grasp_data_dir = os.listdir(f"{DATA_BASE_PATH}/dexgraspnet")
    cnt_scales = [0, 0, 0, 0, 0, 0]
    for file_idx, fname in enumerate(grasp_data_dir, start=1):

        hand_pose_file_path = os.path.join(f"{DATA_BASE_PATH}/dexgraspnet", fname)
        hand_poses = np.load(hand_pose_file_path, allow_pickle=True)

        scales = []

        for hand_pose in hand_poses:
            scales.append(hand_pose['scale'])

        scales = np.array(scales)
        scales = np.unique(scales)
        
        if len(scales) != 5:
            result_cnt += 1
        cnt_scales[len(scales)] += 1
        print(f"{file_idx:05d}/{len(grasp_data_dir):05d}")
    
    print(f"HO pair which doesn't have 5 scales: {result_cnt:04d}")
    print(f"statisitics:")
    for idx in [i for i in range(1, 5+1)]:
        print(f"{idx} = {cnt_scales[idx]}") 

if __name__ == '__main__':
    main()