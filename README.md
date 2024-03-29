# Cross-to-merge training with class balance strategy for learning with noisy labels

<h5 align="center">

*Qian Zhang, Yi Zhu, Ming Yang, Ge Jin, YingWen Zhu, Qiu Chen*

[![Expert Systems with Applications]](https://doi.org/10.1016/j.eswa.2024.123846)
[![License: MIT]](https://github.com/LanXiaoPang613/C2MT/blob/main/LICENSE)

</h5>

The PyTorch implementation code of the paper, [Cross-to-merge training with class balance strategy for learning with noisy labels](https://doi.org/10.1016/j.eswa.2024.123846).

**Abstract**
The collection of large-scale datasets inevitably introduces noisy labels, leading to a substantial degradation in the performance of deep neural networks (DNNs). Although sample selection is a mainstream method in the field of learning with noisy labels, which aims to mitigate the impact of noisy labels during model training, the testing performance of these methods exhibits significant fluctuations across different noise rates and types. In this paper, we propose Cross-to-Merge Training (**C2MT** ), a novel framework that is insensitive to the prior information in sample selection progress, enhancing model robustness. In practical implementation, using cross-divided training data, two different networks are cross-trained with the co-teaching strategy for several local rounds, subsequently merged into a unified model by performing federated averages on the parameters of two models periodically. Additionally, we introduce a new class balance strategy, named Median Balance Strategy (MBS), during the cross-dividing process, which evenly divides the training data into a labeled subset and an unlabeled subset based on the estimated loss distribution characteristics. Extensive experimental results on both synthetic and real-world datasets demonstrate the effectiveness of C2MT. The Code will be available at: https://github.com/LanXiaoPang613/C2MT

![PLReMix Framework](./img/framework.png)

[//]: # (<img src="./img/framework.tig" alt="PLReMix Framework" style="margin-left: 10px; margin-right: 50px;"/>)

## Installation

```shell
# Please install PyTorch using the official installation instructions (https://pytorch.org/get-started/locally/).
pip install -r requirements.txt
```

## Training

To train on the CIFAR dataset(https://www.cs.toronto.edu/~kriz/cifar.html), run the following command:

```shell
python train_cifar_c2mt.py --r 0.2 --lambda_u 0
python train.py --r 0.4 --noise_mode 'asym' --lambda_u 10 --data_path './data/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
python train.py --r 0.5 --noise_mode 'sym' --lambda_u 25 --data_path './data/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
```

To train on the Animal-10N dataset, run the following command:

```shell
python train.py --num_epochs 60 --lambda_u 0 --data_path './data/Animal-10N' --dataset 'animal10N' --num_class 10
```

<details>
<summary>Animal-10N(https://dm.kaist.ac.kr/datasets/animal-10n/) dataset (You need to download the dataset from the corresponding website.)</summary>


## Citation

If you have any questions, do not hesitate to contact zhangqian@jsou.edu.cn

Also, if you find our work useful please consider citing our work:

```bibtex
Qian Zhang, Yi Zhu, Ming Yang, Ge Jin, YingWen Zhu, Qiu Chen,
Cross-to-merge training with class balance strategy for learning with noisy labels,
Expert Systems with Applications,
2024,
123846,
ISSN 0957-4174,
https://doi.org/10.1016/j.eswa.2024.123846.
```

## Acknowledgement

* [DivideMix](https://github.com/LiJunnan1992/DivideMix): The algorithm that our framework is based on.
* [MOIT](https://github.com/DiegoOrtego/LabelNoiseMOIT): Inspiration for the balancing strategy.
* [Federated-Learning](https://github.com/AshwinRJ/Federated-Learning-PyTorch): Inspiration for the cross-to-merge training strategy.
* [Co-teaching]((https://github.com/bhanML/Co-teaching)): Inspiration for the cross-to-merge training strategy.
