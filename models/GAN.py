import torch
from utils import *
import torch.nn as nn
import torch.optim as optim
from models.base_model import base


class Generator(nn.Module):
    def __init__(self, noise_dim, img_size):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
                    self.make_layer(in_cha=noise_dim, out_cha=128),
                    self.make_layer(in_cha=128, out_cha=256),
                    self.make_layer(in_cha=256, out_cha=64),
                    self.make_layer(in_cha=64, out_cha=1024),
                    self.make_layer(in_cha=1024, out_cha=img_size, type=False),
                    nn.Tanh())

    def forward(self, input):
        return self.gen(input)

    @staticmethod
    def make_layer(in_cha, out_cha, type=True):
        """
        This function use to make layer of Generator, last layer don't need LeakyReLU or Batchnorm.
        """
        layer = []
        layer += [nn.Linear(in_features=in_cha, out_features=out_cha)]

        if type is True:
            layer += [nn.ReLU(inplace=True)]

        return nn.Sequential(*layer)


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
                    self.make_layer(in_cha=img_size, out_cha=64),
                    self.make_layer(in_cha=64, out_cha=256),
                    self.make_layer(in_cha=256, out_cha=128),
                    self.make_layer(in_cha=128, out_cha=1, type=False),
                    nn.Sigmoid())

    def forward(self, input):
        return self.disc(input)

    @staticmethod
    def make_layer(in_cha, out_cha, type=True):
        """
        This function use to make layer of Generator, last layer don't need LeakyReLU or Batchnorm.
        """
        layer = []
        layer += [nn.Linear(in_features=in_cha, out_features=out_cha)]

        if type is True:
            layer += [nn.BatchNorm1d(num_features=out_cha)]
            layer += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        return nn.Sequential(*layer)


class GANModel(base):
    def __init__(self, opt):
        """
        This class use to build GAN with Generator and Discriminator.
        """
        super(GANModel, self).__init__(opt)
        self.opt = opt
        self.img_size = opt.resize * opt.resize

        self.image_name = ['Real', 'Fake']
        self.loss_name = ['loss_G', 'loss_D']
        self.model_name = ['G', 'D']

        self.noise_dim = 64
        self.fixed_noise = torch.randn(opt.batch_size, self.noise_dim).to(opt.device)
        self.G = Generator(noise_dim=self.noise_dim, img_size=self.img_size).to(opt.device)
        self.D = Discriminator(img_size=self.img_size).to(opt.device)
        self.adversarial_loss = nn.BCELoss()

        self.get_net = self.get_network(self.model_name, net=(self.G, self.D))

        self.optimize_D = optim.Adam(self.D.parameters(), lr=opt.lr)
        self.optimize_G = optim.Adam(self.G.parameters(), lr=opt.lr)

    def set_input(self, input, label = None):
        self.real = input.view(input.size(0),-1).to(self.opt.device)
        self.noise = torch.randn(input.size(0), self.noise_dim).to(self.opt.device)

    def forward(self):
        self.fake = self.G(self.noise)

    def backward_D(self):
        """
        This function use to build calculate the loss of Discriminator.
        """
        # Real
        pred_real = self.D(self.real)
        self.loss_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real))

        # Fake
        pred_fake = self.D(self.fake)
        self.loss_fake = self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake))

        loss_D = (self.loss_real + self.loss_fake) * 0.5
        loss_D.backward(retain_graph=True)
        return loss_D

    def backward_G(self):
        """
        This function use to build calculate the loss of Generator.
        """
        pred_fake = self.D(self.fake)
        loss_G = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake))
        loss_G.backward()
        return loss_G

    def optimize_parameters(self, batch_idx):
        """
        This function combine of Genrator loss, Discriminator loss and optimizer step for one iteration.
        """
        self.loss_D = 0
        self.loss_G = 0

        self.forward()

        # Discriminator
        if (batch_idx + 1) % 5 == 0: # 5 iterations update Discriminator 1 time
            self.set_requires_grad([self.G], False)
            self.optimize_D.zero_grad()
            self.loss_D = self.backward_D().item()
            self.optimize_D.step()

        # Genrator
        self.set_requires_grad([self.G], True)
        self.optimize_G.zero_grad()
        self.loss_G = self.backward_G().item()
        self.optimize_G.step()
        return [self.loss_G, self.loss_D]

    def evaluate_model(self):
        with torch.no_grad():
          real = self.real.reshape(-1, self.opt.c, self.opt.resize, self.opt.resize)
          fake = self.G(self.fixed_noise).reshape(-1, self.opt.c, self.opt.resize, self.opt.resize)
          return [real[0:25], fake[0:25]]
