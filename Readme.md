# Affinity Feature Strengthening for Accurate, Complete and Robust Vessel Segmentation

## Introduction
Pytorch implementation of the paper ["Affinity Feature Strengthening for Accurate, Complete and Robust Vessel Segmentation", Accepted by JBHI]().  In this paper, we present a novel approach, the affinity feature strengthening network (AFN), which jointly models geometry and refines pixel-wise segmentation features using a contrast-insensitive, multiscale affinity approach. AFN outperforms the state-of-the-art methods in terms of both higher accuracy and topological metrics, while also being more robust to various contrast changes.

## Citation
Please cite the related works in your publications if it helps your research:
comming soon...

```
@ARTICLE{10122604,
  author={Shi, Tianyi and Ding, Xiaohuan and Zhou, Wei and Pan, Feng and Yan, Zengqiang and Bai, Xiang and Yang, Xin},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Affinity Feature Strengthening for Accurate, Complete and Robust Vessel Segmentation}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/JBHI.2023.3274789}}
```

### Prerequisities
* Datasets: [[DRIVE]](https://drive.grand-challenge.org/), [[XCAD]](https://github.com/aisigsjtu/ssvs)
## Usage

#### 1. Training scripts

```bash

sh train.sh

```

#### 2. Evaluation scripts

```bash

sh test.sh

```

## Contact

Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- shitianyihust@hust.edu.cn
- dingxiaohuan@hust.edu.cn
