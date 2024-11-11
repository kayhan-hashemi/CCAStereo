# CCAStereo: Cross-Attention Integration of Contextual and Geometric Features for Efficient Stereo Matching

## Overview
The proposed efficient cost volume-based stereo matching network integrates a Context Cross Attention (CCA) module, which acts as a guidance mechanism in the cost volume aggregation process.

## Performance Comparison on KITTI Benchmarks

| Method | KITTI 2012 |  |  |  | KITTI 2015 |  |  | Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  | 3-noc | 3-all | 4-noc | 4-all | D1-bg | D1-fg | D1-all | (ms) |
| DispNetC [1] | 4.11 | 4.65 | 2.77 | 3.20 | 4.32 | 4.41 | 4.34 | 60 |
| DeepPruneFast [17] | - | - | - | - | 2.32 | 3.91 | 2.59 | 51 |
| AANet [25] | 1.91 | 2.42 | 1.46 | 1.87 | 1.99 | 5.39 | 2.55 | 62 |
| DecNet [26] | - | - | - | - | 2.07 | 3.87 | 2.37 | 50 |
| BGNet+ [11] | 1.62 | 2.03 | 1.16 | 1.48 | 1.81 | 4.09 | 2.19 | 35 |
| CoEx [6] | 1.55 | 1.93 | 1.15 | 1.42 | 1.79 | 3.82 | 2.13 | 33 |
| HITNet [5] | 1.41 | 1.89 | 1.14 | 1.53 | 1.74 | 3.20 | 1.98 | 54 |
| Fast-ACVNet [12] | 1.68 | 2.13 | 1.23 | 1.56 | 1.82 | 3.93 | 2.17 | 43 |
| CGI-Stereo [23] | 1.41 | 1.76 | 1.05 | 1.30 | 1.66 | 3.38 | 1.98 | 36 |
| **Proposed Model** | **1.31** | **1.64** | **0.97** | **1.21** | **1.58** | 3.81 | **1.95** | 57 |

*Note: Lower values indicate better performance. Best results are shown in bold.*

## Prerequisites

### Hardware Requirements
- NVIDIA RTX 3090 or equivalent GPU

### Software Requirements
- Python 3.8 or higher
- PyTorch 1.12 or higher

## Installation

## Directory Structure
Before running the project, please ensure you have the following folder structure set up in the root directory:

```
.
├── pretrained/           # Contains pretrained model files
├── outputs/             # Contains output results
│   ├── eth3D/          # ETH3D dataset results
│   ├── kitti/          # KITTI dataset results
│   ├── middlebury/     # Middlebury dataset results
│   └── sceneflow/      # SceneFlow dataset results
└── checkpoints/        # Contains model checkpoints
    ├── kitti/          # KITTI model checkpoints
    └── sceneflow/      # SceneFlow model checkpoints
```

### Setting up the Environment
```bash
conda create -n venv python=3.8
conda activate venv
```

### Required Dependencies
```bash
# PyTorch and related packages
conda install pytorch torchvision torchaudio

# Additional packages
pip install einops opacus thop open3d pandas opencv-python scikit-image tensorboardx matplotlib tqdm timm
```

## Datasets
The model has been tested with the following benchmark datasets:
- [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
- [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [Middlebury](https://vision.middlebury.edu/stereo/submit3/)

## Training Instructions

### Scene Flow Training
To train CCAStereo on the Scene Flow dataset:
```bash
python train_sceneflow.py --datapath /path/to/sceneflow/dataset
```

### KITTI Training
To fine-tune the model on KITTI using a Scene Flow pretrained model:
```bash
python train_kitti.py \
    --datapath_12 /path/to/kitti2012/dataset \
    --datapath_15 /path/to/kitti2015/dataset \
    --loadckpt ./checkpoints/sceneflow/checkpoint_000099.ckpt
```

## Pretrained Models
Pretrained weights are available for use and evaluation:
- [SceneFlow and KITTI Pretrained Models](https://drive.google.com/file/d/1TO9aIcLdlNWp2hdLyHzXGOWDUfuexCFb/view?usp=sharing)
Note: Place all pretrained model files in `pretrained` folder

## Evaluation

### Cross-Dataset Generalization
Evaluate the model's generalization capabilities across different datasets:

#### KITTI Evaluation
```bash
# KITTI 2015
python test_kitti.py --datapath /path/to/kitti2015/dataset --kitti 2015 --loadckpt ./pretrained/sceneflow.ckpt

# KITTI 2012
python test_kitti.py --datapath /path/to/kitti2012/dataset --kitti 2012 --loadckpt ./pretrained/sceneflow.ckpt
```

#### Middlebury Evaluation
```bash
python test_middlebury.py --datapath /path/to/middlebury/dataset --loadckpt ./pretrained/sceneflow.ckpt
```

#### ETH3D Evaluation
```bash
python test_eth3d.py --datapath /path/to/eth3d/dataset --loadckpt ./pretrained/sceneflow.ckpt
```

## Result Generation and Visualization

### Generate KITTI Predictions
```bash
# KITTI 2015
python save_disp_kitti.py \
    --datapath /path/to/kitti2015/dataset \
    --kitti 2015 \
    --loadckpt ./pretrained/kitti.ckpt \
    --testlist ./filenames/kitti15_test.txt

# KITTI 2012
python save_disp_kitti.py \
    --datapath /path/to/kitti2012/dataset \
    --kitti 2012 \
    --loadckpt ./pretrained/kitti.ckpt \
    --testlist ./filenames/kitti12_test.txt
```

### 3D Point Cloud Visualization
To visualize KITTI results as 3D point clouds:
```bash
python plot_3D.py --datapath /path/to/kitti2015/dataset --loadckpt ./pretrained/kitti.ckpt
```
