# ReID-Survey
Deep Learning for Person Re-identification:  A Survey and Outlook 


# Highlights

- A comprehensive survey and in-depth analysis is conducted for person Re-ID in recent years (2016-2019)

- A new evaluation metric, namely mean Inverse Negative Penalty (mINP), which measures the ability to find the hardest correct match.

- A new AGW baseline with non-local Attention block, Generalized mean pooling and Weighted triplet regularization. It acheieves competitive performance on both single-modality and [cross-modality](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) Re-ID tasks.


# SOTA on Single-Modality Re-ID with mINP

## DukeMTMC

|Method    | Pretrained| Rank@1  | mAP |  mINP |  Model| Paper
| --------   | -----    | -----  |  -----  | ----- |------|------|
|BNNneck     | ImageNet | 86.4% | 76.4%|  40.7% |[Code](https://github.com/michuanhaohao/reid-strong-baseline) |Bag of Tricks and A Strong Baseline for Deep Person Re-identification. In ArXiv 19. [PDF](https://arxiv.org/abs/1903.07071)|
|ABD-Net     | ImageNet | 89.0% | 78.6%| 42.1% | [Code](https://github.com/TAMU-VITA/ABD-Net) |ABD-Net: Attentive but Diverse Person Re-Identification. In ICCV 19. [PDF](https://arxiv.org/abs/1908.01114)|
|AGW     | ImageNet | 89.0%  | 79.6% | 45.7% | [GoogleDrive](https://drive.google.com/open?id=181K9PQGnej0K5xNX9DRBDPAf3K9JosYk)| Deep Learning for Person Re-identification:  A Survey and Outlook  |

## Market-1501

|Method    | Pretrained| Rank@1  | mAP |  mINP |  Model| Paper
| --------   | -----    | -----  |  -----  | ----- |------|------|
|BNNneck     | ImageNet | 94.5% | 85.9%|  59.4% |[Code](https://github.com/michuanhaohao/reid-strong-baseline) |Bag of Tricks and A Strong Baseline for Deep Person Re-identification. In ArXiv 19. [arXiv](https://arxiv.org/abs/1903.07071)|
|ABD-Net     | ImageNet | 95.6% | 88.3%|  66.2% | [Code](https://github.com/TAMU-VITA/ABD-Net) |ABD-Net: Attentive but Diverse Person Re-Identification. In ICCV 19. [PDF](https://arxiv.org/abs/1908.01114)|
|AGW     | ImageNet | 95.1%  | 87.8% | 65.0% | [GoogleDrive](https://drive.google.com/open?id=181K9PQGnej0K5xNX9DRBDPAf3K9JosYk)| Deep Learning for Person Re-identification:  A Survey and Outlook  |
