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

<h2> :computer: Test </h2>

```
python main.py --model "GAN" --phase "test" --epoch "your/epoch/for/loading/generator"
```

Generated images can be seen with `Tensorboard`with `--log_dir=./logs` by default.
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :pushpin: Note </h2>

Currently support  `--dataset=[mnist]` for some basic GAN models. Additional datasets and models will arrive in the future. 

You can build your custom dataset in `dataset.py`.
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :trophy: Generated Images</h2>

<ol>
<li> <b>GAN:</b> 100 epochs</li>
<p align="center">
  <img src="image/GAN/gan.gif" width=400>
</p>

<li> <b>DCGAN:</b> 20 epochs</li>
<p align="center">
  <img src="image/DCGAN/dcgan.gif" width=400>
</p>

<li> <b>cCGAN:</b> 20 epochs</li>
<p align="center">
  <img src="image/cGAN/cgan.gif" width=400>
</p>

</ol>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :bookmark: Models </h2>

Status | Model
:-:| :-:
:heavy_check_mark:| [GAN](https://arxiv.org/abs/1406.2661)
:heavy_check_mark:| [DCGAN](https://arxiv.org/abs/1511.06434)
:heavy_check_mark:| [WGAN](https://arxiv.org/abs/1701.07875)
:heavy_check_mark:| [cGAN](https://arxiv.org/abs/1411.1784)
...| ...
