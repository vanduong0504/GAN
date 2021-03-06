import torch
from utils import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


class Discriminator(nn.Module):
    def __init__(self, channel):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
                    self.make_layer(channel, 16),
                    self.make_layer(16, 32),
                    self.make_layer(32, 64),
                    self.make_layer(64, 1, False),
                    nn.Sigmoid())

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
        self.loss_name = ['loss_G', 'loss_D']
        self.model_name = ['G', 'D']

        self.noise_dim = 64
        self.fixed_noise = torch.randn(opt.batch_size, self.noise_dim).to(opt.device)
        self.G = Generator(self.noise_dim + self.opt.num_class, self.opt.c).to(opt.device)
        self.D = Discriminator(self.opt.c + self.opt.num_class).to(opt.device)
        self.adversarial_loss = nn.BCELoss()

        self.get_net = self.get_network(self.model_name, net=(self.G, self.D))

        self.optimize_D = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(0.5,0.999))
        self.optimize_G = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(0.5,0.999))

    def set_input(self, input, label):
        self.real = input.to(self.opt.device)
        self.label = F.one_hot(label.to(self.opt.device), num_classes=self.opt.num_class)
        noise = torch.randn(input.size(0), self.noise_dim, device=self.opt.device)
        self.noise = torch.cat((noise, self.label), 1)[:,:, None, None].to(self.opt.device)

    def forward(self):
        self.fake = self.G(self.noise)

    def backward_D(self):
        """
        This function use to build calculate the loss of Discriminator.
        """
        image_one_hot_labels = self.label[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, self.real.size(2), self.real.size(3))

        # Real
        self.real_image_and_labels = torch.cat((self.real, image_one_hot_labels), 1)
        pred_real = self.D(self.real_image_and_labels)
        self.loss_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real))

        # Fake
        self.fake_image_and_labels = torch.cat((self.fake, image_one_hot_labels), 1)
        pred_fake = self.D(self.fake_image_and_labels)
        self.loss_fake = self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake))

        loss_D = (self.loss_real + self.loss_fake) * 0.5
        loss_D.backward(retain_graph=True)
        return loss_D

    def backward_G(self):
        """
        This function use to build calculate the loss of Generator.
        """
        pred_fake = self.D(self.fake_image_and_labels)
        loss_G = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake))
        loss_G.backward()
        return loss_G

    def optimize_parameters(self):
        """
        This function combine of Genrator loss, Discriminator loss and optimizer step for one iteration.
        """
        self.forward()

        # Discriminator
        self.set_requires_grad([self.G], False)
        self.optimize_D.zero_grad()
        loss_D = self.backward_D().item()
        self.optimize_D.step()

        # Genrator
        self.set_requires_grad([self.G], True)
        self.optimize_G.zero_grad()
        loss_G = self.backward_G().item()
        self.optimize_G.step()
        return [loss_G, loss_D]

    def evaluate_model(self):
        with torch.no_grad():
          fixed_noise_with_label = torch.cat((self.fixed_noise, self.label), 1)[:,:, None, None]
          fake = self.G(fixed_noise_with_label)
          return [self.real[0:25], fake[0:25]]
