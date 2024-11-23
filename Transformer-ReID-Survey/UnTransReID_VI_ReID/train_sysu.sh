#!/bin/bash
#SBATCH -A chenjun3
#SBATCH -p a100x4
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --exclude=g0154,g0150,g0158
#SBATCH -o sysu_2p_384_g.log
# module load scl/gcc5.3
module load nvidia/cuda/11.6
# rm -rf x.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3 python cluster_contrast_camera_cmrouting.py -b 256 -a agw -d  sysu_all --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
CUDA_VISIBLE_DEVICES=0,1 python sysu_train.py -b 256 -a agw -d  sysu_all --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
