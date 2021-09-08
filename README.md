<h1 align="center"> GENERATIVE ADVERSARIAL NETWORKS </h1>

[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<h2> :pencil: About the project </h2>

This project aims to generate images with GAN models. I will use **MNIST** and **Fashion MNIST** datasets for *GAN*, *cGAN*, and *DCGAN*. 
Other datasets depend on different tasks, e.g., **horsetozibra** for *CycleGAN*.
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :computer: Train </h2>

```
python main.py --model "GAN" --phase "train"
```
Generate image can be seen with `Tensorboard`with `--log_dir=./logs` by default.
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :pushpin: Note </h2>

Currently support  `--dataset=[mnist, fashionmnist]` for some basic GAN models and `--phase=train`. Additional datasets and models will arrive in the future. 

You can build your custom dataset in `dataset.py`.
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :bookmark: Models </h2>

Status | Model
:-:| :-:
:heavy_check_mark:| [GAN](https://arxiv.org/abs/1406.2661)
:x:| [cGAN](https://arxiv.org/abs/1411.1784)
:x:| [DCGAN](https://arxiv.org/abs/1511.06434)
...| ...
