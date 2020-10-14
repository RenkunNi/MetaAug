# Data Augmentation for Meta-Learning

### Abstract

Conventional image classifiers are trained by randomly sampling mini-batches of images. To achieve state-of-the-art performance, sophisticated data augmentation schemes are used to expand the amount of training data available for sampling. In contrast, meta-learning algorithms sample not only images, but classes as well. We investigate how data augmentation can be used not only to expand the number of images available per class, but also to generate entirely new classes. We systematically dissect the meta-learning pipeline and investigate the distinct ways in which data augmentation can be integrated at both the image and class levels. Our proposed meta-specific data augmentation significantly improves the performance of meta-learners on few-shot classification benchmarks.

## Dependencies
* Python 3.6+
* [PyTorch 0.4.0+](http://pytorch.org)
* [qpth 0.0.11+](https://github.com/locuslab/qpth)
* [tqdm](https://github.com/tqdm/tqdm)
* [kornia](https://github.com/kornia/kornia)

## Usage

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/RenkunNi/MetaAug.git
    cd MetaAug
    ```
2. Download and decompress dataset files: [**miniImageNet**](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (courtesy of [**Spyros Gidaris**](https://github.com/gidariss/FewShotWithoutForgetting)), [**CIFAR-FS**](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)


### Meta-training Examples

1. To train with data augmentations (i.e. query cutmix and task large rotation) on 5-way CIFAR-FS:

    ```bash
    python train_aug.py --gpu 0 --save-path "./experiments/ResNet_R2D2_qcm_tlr" --train-shot 5 \
    --head R2D2 --network ResNet --dataset CIFAR_FS --query_aug cutmix --s_p 1. --task_aug Rot90 --t_p 0.25
    ```

3. To train Meta-MaxUp (4 samples) on 5-way CIFAR-FS:
    ```bash
    python train_maxup.py --gpu 0,1,2,3 --save-path "./experiments/ResNet_R2D2_maxup_4" --train-shot 5 \
    --head R2D2 --network ResNet --dataset CIFAR_FS --m 4
    ```
### Meta-testing Examples
1. To test models on 5-way-N-shot CIFAR-FS:
```
python test.py --gpu 0 --load ./experiments/ResNet_R2D2_maxup_4/best_model.pth --episode 1000 \
--way 5 --shot N --query 15 --head R2D2 --network ResNet --dataset CIFAR_FS 
```
2. To test models on 5-way-N-shot CIFAR-FS with shot augmentation (flip):
```
python test.py --gpu 0 --load ./experiments/ResNet_R2D2_maxup_4/best_model.pth --episode 1000 \
--way 5 --shot N --query 15 --head R2D2 --network ResNet --dataset CIFAR_FS --shot_aug fliplr --s_du 2
```
3. To test models on 5-way-N-shot CIFAR-FS with shot augmentation (flip) and ensemble:
```
python test_ens.py --gpu 0 --load ./experiments/ResNet_R2D2_maxup_4/ --episode 1000 \
--way 5 --shot N --query 15 --head R2D2 --network ResNet --dataset CIFAR_FS --shot_aug fliplr --s_du 2
```
## Acknowledgments

This code is based on the implementations of [**Prototypical Networks**](https://github.com/cyvius96/prototypical-network-pytorch),  [**Dynamic Few-Shot Visual Learning without Forgetting**](https://github.com/gidariss/FewShotWithoutForgetting), [**DropBlock**](https://github.com/miguelvr/dropblock), [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet) and [**TaskLevelAug**](https://github.com/AceChuse/TaskLevelAug).
