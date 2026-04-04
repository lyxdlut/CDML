# First-Order Cross-Domain Meta Learning for Few-Shot Remote Sensing Object Classification
This repository includes introductions and implementation of ***First-Order Cross-Domain Meta Learning for Few-Shot Remote Sensing Object Classification*** in PyTorch.

# Datasets
We conduct experiments using a benchmark covering multiple modalities, utilizing five optical remote sensing datasets: 
***[Aerial Image Dataset (Xia et al. 2017)](https://captain-whu.github.io/AID/), [NWPU-RESISC45 (Cheng et al. 2017)](https://gcheng-nwpu.github.io/), 
[RSI-CB256 (Li et al. 2019)](https://github.com/lehaifeng/RSI-CB), [UC Merced Land Use (Yang et al. 2010)](https://aistudio.baidu.com/datasetdetail/51628) and [PatternNet (Zhou et al. 2018)](https://sites.google.com/view/zhouwx/dataset)***, alongside newly constructed ***infrared dataset and SAR dataset***. The ***infrared dataset*** is compiled from images sourced from ***[MRSSC2.0](https://csu.cas.cn/gb/kybm/sjlyzx/gcxx_sjj/sjj_tgxl/202208/t20220831_6507453.html), [DroneVehicle](https://github.com/VisDrone/DroneVehicle), and [VEDAI](https://downloads.greyc.fr/vedai/)***, while the ***SAR dataset*** integrates images from ***[MRSSC2.0](https://csu.cas.cn/gb/kybm/sjlyzx/gcxx_sjj/sjj_tgxl/202208/t20220831_6507453.html), [SARDet-100K](github.com/zcablii/SARDet_100K), [BRIGHT](https://github.com/ChenHongruixuan/BRIGHT), and [FUSAR-Ship1.0](https://noda.ac.cn/datasharing/datasetDetails/659b92d99c863b5bffce9c0d)***. We reorganized the dataset categories to ensure that the categories in the test domain remain unseen during the training phase.

# File Structure
## 1. Model Download
You can download the required model files (`model_path.zip`) via Baidu Netdisk:
* **Link:**[https://pan.baidu.com/s/1KBosdFMZbBQKuKpaJAuNvg](https://pan.baidu.com/s/1KBosdFMZbBQKuKpaJAuNvg)
* **Extraction Code / Password:** `dlut`

Once downloaded and extracted, please place the model files in the following directory:
> `outputs/CDML_meta/`

## 2. Dataset Preparation
Please place your prepared dataset files directly into the `datasets/` directory.
```
CDML
.
├── dataset
│   ├── AID
│   ├── IR
│   ├── NWPU
│   ├── PatternNet
│   ├── RSI
│   ├── SAR
│   ├── UC
│   ├── my_utils
│   │   └── spearman.py
│   ├── Resnet
│   │   └── model.py
│   ├── train.py
│   └── VGG
│       └── model.py
├── outputs
│   ├── CDML_meta
│   ├── pretrain_LAT
├── LAT_utils.py
├── Option.py
├── ProtoNet.py
├── README.md
├── config.py
├── learner.py
├── requirements.txt
├── test.py
├── test_dataset.py
├── train.py
├── train_dataset.py
└──utils.py

```

# Requirements

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) to manage your environment. Follow the steps below to set up the required dependencies.

## 1. Create a Conda Environment
First, create a new conda environment and activate it. You can replace `myenv` with your preferred environment name, and adjust the Python version if necessary:

```bash
conda create -n myenv python=3.9 -y
conda activate myenv

```

## 2. Install Dependencies
Once the environment is activated, install all the required packages using the requirements.txt file provided in the repository:

```bash
pip install -r requirements.txt
```

# Train and Eval

## Train
To train the model, simply run the `train.py` script:

```bash
python train.py
```

## Eval
To evaluate the model's performance on the test dataset, you can run the test.py script:

```bash
python test.py
```
