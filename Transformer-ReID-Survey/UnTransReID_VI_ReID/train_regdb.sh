#!/bin/bash
#SBATCH -A chenjun3
#SBATCH -p a100x4
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH -o regdb_2p_384_g.log
module load nvidia/cuda/11.6

for trial in 1 2 3 4 5 6 7 8 9 10
do
CUDA_VISIBLE_DEVICES=4,5,6,7 python regdb_train.py -b 256 -a agw -d  regdb_rgb --iters 50 --momentum 0.95 --eps 0.6 --num-instances 16 --trial $trial
done
echo 'Done!'
#cluster_contrast_camera_cmsub_andcmass_regdb.py

# CUDA_VISIBLE_DEVICES=0,1,2,3 python cluster_contrast_camera_cmsub_andcmass_regdb.py -b 256 -a agw -d  regdb_all --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python cluster_contrast_camera_cmsub_s3mergecamera.py -b 256 -a agw -d  sysu_all --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 


# CUDA_VISIBLE_DEVICES=0,1,2,3 python cluster_contrast_camera.py -b 256 -a agw -d  sysu_all --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16


# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16


# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --height 224 --width 224
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16 --height 224 --width 224

