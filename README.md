# **IMIS-Benchmark**

[![PyTorch](https://img.shields.io/badge/HuggingFace-DataSet-FFD21E?logo=huggingface)](https://huggingface.co/datasets/General-Medical-AI/IMed-361M)
[![PyTorch](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](https://pytorch.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![GitHub repo size](https://img.shields.io/github/repo-size/tuhlnaa/IMIS-Bench-Extended?label=Repo%20size)](https://github.com/tuhlnaa/IMIS-Bench-Extended)

<br>

## Abstract

[![Homepage](https://img.shields.io/badge/GitHub-Homepage-181717?logo=github)](https://uni-medical.github.io/IMIS-Benchmark)
[![arXiv](https://img.shields.io/badge/arXiv-2411.12814-B31B1B?logo=arxiv)](https://arxiv.org/abs/2411.12814)

This repository hosts the code and resources for the paper **"Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline"**.

We collected 110 medical image datasets from various sources and generated the **IMed-361M** dataset, which contains over **361 million masks**, through a rigorous and standardized data processing pipeline. Using this dataset, we developed the **IMIS baseline network**.

<p align="center">
    <img width="1000" alt="image" src="https://github.com/uni-medical/IMIS-Bench/blob/main/assets/fig1.png">
</p>

<br>

### IMIS Benchmark Dataset: IMed-361M

The IMed-361M dataset is the largest publicly available multimodal interactive medical image segmentation dataset, featuring **6.4 million images**, **273.4 million masks** (56 masks per image), **14 imaging modalities**, and **204 segmentation targets**. It ensures diversity across six anatomical groups, fine-grained annotations with most masks covering <2% of the image area, and broad applicability with 83% of images in resolutions between 256×256 and 1024×1024. IMed-361M offers 14.4 times more masks than MedTrinity-25M, significantly surpassing other datasets in scale and mask quantity.
<p align="center"><img width="800" alt="image" src="https://github.com/uni-medical/IMIS-Bench/blob/main/assets/fig2.png"></p> 

<br>

### IMIS Network

We simulate continuous interactive segmentation training.
<p align="center"><img width="800" alt="image" src="https://github.com/uni-medical/IMIS-Bench/blob/main/assets/fig4.png"></p> 

<br>

## Installation
```sh
git clone https://github.com/uni-medical/IMIS-Bench.git
```

<br>

## Environment Setup
The recommended operating environment is as follows:

| Package           | Version    | Package         | Version |
|-------------------|------------|-----------------|---------|
| CUDA              | 11.8       | timm            | 0.9.16  |
| Huggingface-Hub   | 0.23.4     | transformers    | 4.39.3  |
| nibabel           | 5.2.1      | monai           | 0.9.1   |
| Python            | 3.8.19     | opencv-python   | 4.10.0  |
| PyTorch           | 2.2.1      | torchvision     | 0.17.2  |

<br>

## Datasets
IMed-361 was created by preprocessing a combination of private and publicly available medical image segmentation datasets. The dataset will be made available on [HuggingFace](https://huggingface.co/datasets/1Junlong/IMed-361M/tree/main). For detailed information about the source datasets, please refer to our [paper](https://arxiv.org/pdf/2411.12814). To help you get started quickly, we have provided a small sample demonstration IMIS-Bench/dataset from IMed-361.

```sh
dataset
├── BTCV
│    ├─ image
│    │    ├── xxx.png
│    │    ├── ....
│    │    ├── xxx.png
│    ├── label
│    │    ├── xxx.npz
│    │    ├── ....
│    │    ├── xxx.npz
│    ├── imask
│    │    ├── xxx.npy
│    │    ├── ....
│    │    ├── xxx.npy
│    └── dataset.json
```

<br>

## Model Checkpoints

We host our model checkpoints on Baidu Netdisk: https://pan.baidu.com/s/1eCuHs3qhd1lyVGqUOdaeFw?pwd=r1pg, Password：r1pg 

Please download the checkpoint from Baidu Netdisk and place them under **"ckpt/"**.

<br>

## Train IMIS-Net

To train the IMIS-Net, run:
```sh
cd IMIS-Bench
```

```sh
python train.py
```

- work_dir: Specifies the working directory for the training process. Default value is `work_dir`.
- image_size: Default value is 1024.
- mask_num: Specify the number of masks corresponding to one image, with a default value of 5.
- data_path: Dataset directory, for example: `dataset/BTCV`.
- sam_checkpoint: Load our checkpoint.
- inter_num: Mask decoder iterative runs.

<br>

## Evaluate IMIS-Net

To evaluate the IMIS-Net, run:
```sh
python test.py
```
- test_mode: Set to `True`
- image_size: Default value is 1024.
- prompt_mode: Specifies the interaction mode, supporting `points`, `bboxes` and `text`.
- inter_num: Simulate interactive annotation correction times.

<br>

## Citation

Please cite our paper if you use the code, model, or data.

```bibtex
@article{cheng2024interactivemedicalimagesegmentation,
      title={Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline}, 
      author={Junlong Cheng and Bin Fu and Jin Ye and Guoan Wang and Tianbin Li and Haoyu Wang and Ruoyu Li and He Yao and Junren Chen and JingWen Li and Yanzhou Su and Min Zhu and Junjun He},
      year={2024},
      eprint={2411.12814},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.12814}, 
}
```
