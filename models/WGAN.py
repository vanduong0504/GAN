import torch
from utils import *
import torch.nn as nn
import torch.optim as optim
from models.base_model import base


class Generator(nn.Module):
    def __init__(self, noise_dim, channel):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
                    self.make_layer(noise_dim, 512),
                    self.make_layer(512, 256),
                    self.make_layer(256, 128),
                    self.make_layer(128, 64),
                    self.make_layer(64, channel, False),
                    nn.Tanh())

    def forward(self, input):
        return self.gen(input)

    @staticmethod
    def make_layer(in_cha, out_cha, type=True):
        """
        This function use to make layer of Generator, last layer don't need ReLU or Batchnorm.
        """
        layer = []
        layer += [nn.ConvTranspose2d(in_cha, out_cha, *(4,2,1))]

        if type is True:
            layer += [nn.BatchNorm2d(out_cha)]
            layer += [nn.ReLU(True)]

        return nn.Sequential(*layer)


class Critic(nn.Module):
    def __init__(self, channel):
        super(Critic, self).__init__()

        self.disc = nn.Sequential(
                    self.make_layer(channel, 16),
                    self.make_layer(16, 32),
                    self.make_layer(32, 64),
                    self.make_layer(64, 1, False))

    def forward(self, input):
        return self.disc(input).view(-1, 1)

    @staticmethod
    def make_layer(in_cha, out_cha, type=True):
        """
        This function use to make layer of Generator, last layer don't need LeakyReLU or Batchnorm.
        """
        layer = []
        layer += [nn.Conv2d(in_cha, out_cha, *(4,2,1))]

        if type is True:
            layer += [nn.BatchNorm2d(out_cha)]
            layer += [nn.LeakyReLU(0.2, True)]

        return nn.Sequential(*layer)


class Model(base):
    def __init__(self, opt):
        """
        This class use to build GAN with Generator and Discriminator.
        """
        super(Model, self).__init__(opt)
        self.opt = opt

        self.image_name = ['Real', 'Fake']
        self.loss_name = ['loss_G', 'loss_C']
        self.model_name = ['G', 'C']

        self.n_critic = 5
        self.clip_weight = 1e-2

        self.noise_dim = 64
        self.fixed_noise = torch.randn(opt.batch_size, self.noise_dim).to(opt.device)
        self.G = Generator(self.noise_dim, self.opt.c).to(opt.device)
        self.C = Critic(self.opt.c).to(opt.device)

        self.get_net = self.get_network(self.model_name, net=(self.G, self.C))

        self.optimize_C = optim.RMSprop(self.C.parameters(), lr=opt.lr)
        self.optimize_G = optim.RMSprop(self.G.parameters(), lr=opt.lr)

    def set_input(self, input, label):
        self.real = input.to(self.opt.device)
        self.noise = torch.randn(input.size(0), self.noise_dim)[:,:, None, None].to(self.opt.device)

    def forward(self):
        self.fake = self.G(self.noise)

    def backward_D(self):
        """
        This function use to build calculate the loss of Discriminator.
        """
        # Real
        pred_real = self.C(self.real)

        # Fake
        pred_fake = self.C(self.fake)

        loss_C = torch.mean(pred_fake) - torch.mean(pred_real)
        loss_C.backward(retain_graph=True)
        return loss_C

    def backward_G(self):
        """
        This function use to build calculate the loss of Generator.
        """
        pred_fake = self.C(self.fake)
        loss_G = - torch.mean(pred_fake)
        loss_G.backward()
        return loss_G

    def optimize_parameters(self):
        """
        This function combine of Genrator loss, Discriminator loss and optimizer step for one iteration.
        """
        loss_G, loss_C = 0, 0
        self.forward()
        
        # Critic
        for _ in range(self.n_critic):
            self.set_requires_grad([self.G], False)
            self.optimize_C.zero_grad()
            loss_C = self.backward_D().item()
            self.optimize_C.step()
        
            # Clip the weight to range[-cliping_weight, +cliping_weight]
            for param in self.C.parameters():
                param.data.clip_(-self.clip_weight, self.clip_weight)

        # Genrator
        self.set_requires_grad([self.G], True)
        self.optimize_G.zero_grad()
        loss_G = self.backward_G().item()
        self.optimize_G.step()
        
        return [loss_G, loss_C]

    def evaluate_model(self):
        with torch.no_grad():
          fake = self.G(self.fixed_noise[:,:, None, None])
          return [self.real[0:25], fake[0:25]]
