 CUDA_VISIBLE_DEVICES=0,1,2,3 examples/cluster_contrast_train_usl.py -b 256 -a vit_base -d market1501 --iters 200 --eps 0.6 --self-norm --use-hard --hw-ratio 2 --num-instances 8 --conv-stem
