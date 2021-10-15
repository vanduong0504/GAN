import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import OrderedDict
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter


def init_weight(net, name):
    print(f"Weight Initialization for {name}")
    for layer in net.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(layer.weight, 0, 0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.normal_(layer.weight, 0.0, 0.02)
            nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.02)
            nn.init.constant_(layer.bias, 0)


def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def load(net, path):
    net.load_state_dict(torch.load(path))
    return net


def create_logger(save_dir, folder, log):
    """
    This function to create tensorboard events for models.
    """
    write = OrderedDict()
    for item in log:
        write[item] = SummaryWriter(os.path.join(save_dir, folder, item))
    return write


def current_losses(loss_name):
    """
    This function to get the model current loss.
    """
    loss = OrderedDict()
    for item in loss_name:
        loss[item] = []
    return loss


def grid_image(output):
    """
    This function to build grid_image to visualize images in Tensorboard.
    """
    grid = []
    for data in output:
        grid += [make_grid(data, nrow=5, normalize=True)]
    return grid


def save_result(image, dir, epoch=None):
    if epoch is not None:
        save_image(image, f"{dir}/{epoch}.png", nrow=5, normalize=True)
    else:
        save_image(image, f"{dir}/test.png", nrow=int(image.size(0)/8), normalize=True)
        