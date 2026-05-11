#!/bin/bash
#SBATCH --job-name=data_review
#SBATCH --output=/workspace/lab_intern/KDW/GraspRep/logs/slurm_%j.out
#SBATCH --error=/workspace/lab_intern/KDW/GraspRep/logs/slurm_%j.err
#SBATCH --partition=intern
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=intern_qos
#SBATCH --time=2-00:00:00  

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kdw_dexgraspnet

cd /workspace/lab_intern/KDW/GraspRep
mkdir -p ./logs

export PYTHONUNBUFFERED=1
python -u subtasks/01_data_review.py

echo "Job ID: $SLURM_JOB_ID" >> logs/job_info.txt
echo "Node: $SLURM_NODELIST" >> logs/job_info.txt
echo "Completed: $(date)" >> logs/job_info.txt