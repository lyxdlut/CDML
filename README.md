# First-Order Cross-Domain Meta Learning for Few-Shot Remote Sensing Object Classification
This repository includes introductions and implementation of ***First-Order Cross-Domain Meta Learning for Few-Shot Remote Sensing Object Classification*** in PyTorch.

# Datasets
We conduct experiments using a benchmark covering multiple modalities, utilizing five optical remote sensing datasets: 
***[Aerial Image Dataset (Xia et al. 2017)](https://captain-whu.github.io/AID/), [NWPU-RESISC45 (Cheng et al. 2017)](https://gcheng-nwpu.github.io/), 
[RSI-CB256 (Li et al. 2019)](https://github.com/lehaifeng/RSI-CB), [UC Merced Land Use (Yang et al. 2010)](https://aistudio.baidu.com/datasetdetail/51628) and [PatternNet (Zhou et al. 2018)](https://sites.google.com/view/zhouwx/dataset)***, alongside newly constructed ***infrared dataset and SAR dataset***. The ***infrared dataset*** is compiled from images sourced from ***[MRSSC2.0](https://csu.cas.cn/gb/kybm/sjlyzx/gcxx_sjj/sjj_tgxl/202208/t20220831_6507453.html), [DroneVehicle](https://github.com/VisDrone/DroneVehicle), and [VEDAI](https://downloads.greyc.fr/vedai/)***, while the ***SAR dataset*** integrates images from ***[MRSSC2.0](https://csu.cas.cn/gb/kybm/sjlyzx/gcxx_sjj/sjj_tgxl/202208/t20220831_6507453.html), [SARDet-100K](github.com/zcablii/SARDet_100K), [BRIGHT](https://github.com/ChenHongruixuan/BRIGHT), and [FUSAR-Ship1.0](https://noda.ac.cn/datasharing/datasetDetails/659b92d99c863b5bffce9c0d)***. We reorganized the dataset categories to ensure that the categories in the test domain remain unseen during the training phase.

