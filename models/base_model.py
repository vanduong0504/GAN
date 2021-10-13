import os
import torch
from utils import *
from abc import ABC, abstractmethod


class base(ABC):
    """
    This base class for others GAN models
    """
    def __init__(self, opt):
        self.opt = opt
        self.model_name = []

    @abstractmethod
    def set_input(self, input=None, label=None):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self, batch_idx):
        pass

    def train(self):
        for name in self.model_name:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def eval(self):
        for name in self.model_name:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def get_network(self, model_name, net):
        """
        This function return a dictionary where key are self.models_name
        and values a instance of it.
        """
        return OrderedDict({name: net[i] for i, name in enumerate(model_name)})

    def load_networks(self, epoch):
        if self.opt.phase == "train":
            for name in self.model_name:
                net = getattr(self, name, None)
                path = os.path.join(self.opt.save_path, self.opt.model, self.opt.dataset, name, f"{name}_{epoch}.pth")
                load(net, path)
                print(f"Load {name} of {self.opt.model} at epoch {epoch}")
        else:
            for name in self.model_name:
                if name.startswith("G"):
                    net = getattr(self, name, None)
                    path = os.path.join(self.opt.save_path, self.opt.model, self.opt.dataset, name, f"{name}_{epoch}.pth")
                    load(net, path)
                    print(f"Load {name} of {self.opt.model} at epoch {epoch}")
        print("Successfully finish loading!!!")

    def save_networks(self, epoch):
        for name in self.model_name:
            path = os.path.join(self.opt.save_path, self.opt.model, self.opt.dataset, name)
            net = getattr(self, name)
            torch.save(net.state_dict(), check_folder(path) + f"/{name}_{epoch}.pth")

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad