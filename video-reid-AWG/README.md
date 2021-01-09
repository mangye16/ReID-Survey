# AGW Baseline for Video Person ReID

This repository contains PyTorch implementations of AGW Baseline for video-based person reID. 
The code is mainly based on [Video-Person-ReID](https://github.com/jiyanggao/Video-Person-ReID).

## Quick Start

### 1. Prepare dataset 
### Dataset

1. Create a directory named `mars/` under `data/`.
2. Download dataset to `data/mars/` from http://www.liangzheng.com.cn/Project/project_mars.html.
3. Extract `bbox_train.zip` and `bbox_test.zip`.
4. Download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put `info/` in `data/mars` (we want to follow the standard split in [8]). The data structure would look like:
```
mars/
    bbox_test/
    bbox_train/
    info/
```
### 2. Train

To train a AGW+ model on MARS with GPU device 0, run similarly:
```
CUDA_VISIBLE_DEVICES=0 python ./main_video_person_reid.py  --arch AGW_Plus_Baseline \
--train-dataset mars --test-dataset mars  --save-dir ./mars_agw_plus
```

## Citation

Please kindly cite this paper in your publications if it helps your research:
```
@article{arxiv20reidsurvey,
  title={Deep Learning for Person Re-identification: A Survey and Outlook},
  author={Ye, Mang and Shen, Jianbing and Lin, Gaojie and Xiang, Tao and Shao, Ling and Hoi, Steven C. H.},
  journal={arXiv preprint arXiv:2001.04193},
  year={2020},
}
```

Contact: mangye16@gmail.com
