import os
import torch.nn as nn
from collections import OrderedDict
from torchvision.transforms.functional import normalize
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter


def init_weight(net, name):
    print(f"Weight Initialization for {name}")
    for layer in net.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
            nn.init.constant_(layer.bias, 0)


def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def load(net, path):
    net.load_state_dict(path)
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


def save_image(image, dir):
    save_image(image, f"{dir}/image.png", nrow=5, normalize=True)