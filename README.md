# ReID-Survey
Deep Learning for Person Re-identification:  A Survey and Outlook.

A cross-modality implementation is [HERE](https://github.com/mangye16/Cross-Modal-Re-ID-baseline).


## Highlights

- A comprehensive survey with in-depth analysis for person Re-ID in recent years (2016-2019).

- A new evaluation metric, namely mean Inverse Negative Penalty (mINP), which measures the ability to find the hardest correct match.

- A new AGW baseline with non-local Attention block, Generalized mean pooling and Weighted regularization triplet. It acheieves competitive performance on both single-modality and [cross-modality](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) Re-ID tasks.


## AGW on Single-Modality Re-ID with mINP

### DukeMTMC dataset

|Method    | Pretrained| Rank@1  | mAP |  mINP |  Model| Paper
| --------   | -----    | -----  |  -----  | ----- |------|------------------------|
|BNNneck     | ImageNet | 86.4% | 76.4%|  40.7% |[Code](https://github.com/michuanhaohao/reid-strong-baseline) |Bag of Tricks and A Strong Baseline for Deep Person Re-identification. In ArXiv 19. [PDF](https://arxiv.org/abs/1903.07071)|
|ABD-Net     | ImageNet | 89.0% | 78.6%| 42.1% | [Code](https://github.com/TAMU-VITA/ABD-Net) |ABD-Net: Attentive but Diverse Person Re-Identification. In ICCV 19. [PDF](https://arxiv.org/abs/1908.01114)|
|AGW     | ImageNet | 89.0%  | 79.6% | 45.7% | [GoogleDrive](https://drive.google.com/open?id=181K9PQGnej0K5xNX9DRBDPAf3K9JosYk)| Deep Learning for Person Re-identification:  A Survey and Outlook  |

### Market-1501 dataset

|Method    | Pretrained| Rank@1  | mAP |  mINP |  Model| Paper
| --------   | -----    | -----  |  -----  | ----- |------|------|
|BNNneck     | ImageNet | 94.5% | 85.9%|  59.4% |[Code](https://github.com/michuanhaohao/reid-strong-baseline) |Bag of Tricks and A Strong Baseline for Deep Person Re-identification. In ArXiv 19. [arXiv](https://arxiv.org/abs/1903.07071)|
|ABD-Net     | ImageNet | 95.6% | 88.3%|  66.2% | [Code](https://github.com/TAMU-VITA/ABD-Net) |ABD-Net: Attentive but Diverse Person Re-Identification. In ICCV 19. [PDF](https://arxiv.org/abs/1908.01114)|
|AGW     | ImageNet | 95.1%  | 87.8% | 65.0% | [GoogleDrive](https://drive.google.com/open?id=181K9PQGnej0K5xNX9DRBDPAf3K9JosYk)| Deep Learning for Person Re-identification:  A Survey and Outlook  |


### CUHK03 dataset

|Method    | Pretrained| Rank@1  | mAP |  mINP |  Model| Paper
| --------   | -----    | -----  |  -----  | ----- |------|------|
|BNNneck     | ImageNet | 58.0% | 56.6%|  43.8% |[Code](https://github.com/michuanhaohao/reid-strong-baseline) |Bag of Tricks and A Strong Baseline for Deep Person Re-identification. In ArXiv 19. [PDF](https://arxiv.org/abs/1903.07071)|
|AGW     | ImageNet | 63.6%  | 62.0% | 50.3% | [GoogleDrive](https://drive.google.com/open?id=181K9PQGnej0K5xNX9DRBDPAf3K9JosYk)| Deep Learning for Person Re-identification:  A Survey and Outlook  |

### MSMT17 dataset

|Method    | Pretrained| Rank@1  | mAP |  mINP |  Model| Paper
| --------   | -----    | -----  |  -----  | ----- |------|------|
|BNNneck     | ImageNet | 63.4% | 45.1%|  12.4% |[Code](https://github.com/michuanhaohao/reid-strong-baseline) |Bag of Tricks and A Strong Baseline for Deep Person Re-identification. In ArXiv 19. [arXiv](https://arxiv.org/abs/1903.07071)|
|AGW     | ImageNet | 68.3% | 49.3%|  14.7% | [GoogleDrive](https://drive.google.com/open?id=181K9PQGnej0K5xNX9DRBDPAf3K9JosYk)| Deep Learning for Person Re-identification:  A Survey and Outlook  |


### Citation

Please kindly cite this paper in your publications if it helps your research:
```
@inproceedings{arxiv20reidsurvey,
  title={Deep Learning for Person Re-identification: A Survey and Outlook},
  author={Ye, Mang and Shen, Jianbing and Lin, Gaojie and Xiang, Tao and Shao, Ling and Hoi, Steven C. H.},
  booktitle={ArXiv},
  year={2020},
}
```

Contact: mangye16@gmail.com
