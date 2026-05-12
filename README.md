# Grasp Box Representations
This codebase is built upon the following repositories:

* HOGraspNet (ECCV 2024): https://github.com/kaist-uvr-lab/HOGraspNet
* DexGraspNet 
* DexGraspNet_mano (mano branch)

If use cloned repo, please remove <code>.git</code>

# Environments
```bash
# For Surface sampling DexGraspNet
conda create -n kdw_dexgraspnet python=3.7
conda activate kdw_dexgraspnet

pip install torch==1.13.1
pip install transforms3d==0.4.2
pip install trimesh==4.4.1
pip install plotly==5.18.0

cd ~/KDW/GraspRep/DexGraspNet/thirdparty/pytorch_kinematics
pip install -e .

cd ..
cd ..
wget -c -P data https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/dexgraspnet.tar.gz
wget -c -P data https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/meshdata.tar.gz
tar -xvzf data/dexgraspnet.tar.gz -C data
tar -xvzf data/meshdata.tar.gz -C data
```