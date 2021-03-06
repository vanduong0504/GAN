import torch
from utils import *
import torch.nn as nn
import torch.optim as optim
from models.base_model import base


class Generator(nn.Module):
    def __init__(self, noise_dim, img_size):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
                    self.make_layer(noise_dim, 256),
                    self.make_layer(256, 512),
                    self.make_layer(512, 1024),
                    self.make_layer(1024, img_size, False),
                    nn.Tanh())    

    def forward(self, input):
        return self.gen(input)

    @staticmethod
    def make_layer(in_cha, out_cha, type=True):
        """
        This function use to make layer of Generator, last layer don't need LeakyReLU.
        """
        layer = []
        layer += [nn.Linear(in_cha, out_cha)]

        if type is True:
            layer += [nn.LeakyReLU(0.2)]

        return nn.Sequential(*layer)


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
                    self.make_layer(img_size, 1024),
                    self.make_layer(1024, 512),
                    self.make_layer(512, 256),
                    self.make_layer(256, 1, False),
                    nn.Sigmoid())

    def forward(self, input):
        return self.disc(input).view(-1, 1)
        
    @staticmethod
    def make_layer(in_cha, out_cha, type=True):
        """
        This function use to make layer of Generator, last layer don't need LeakyReLU.
        """
        layer = []
        layer += [nn.Linear(in_cha, out_cha)]

        if type is True:
            layer += [nn.LeakyReLU(0.2)]
            layer += [nn.Dropout(0.3)]

        return nn.Sequential(*layer)


class Model(base):
    def __init__(self, opt):
        """
        This class use to build GAN with Generator and Discriminator.
        """
        super(Model, self).__init__(opt)
        self.opt = opt
        self.img_size = opt.resize * opt.resize

        self.image_name = ['Real', 'Fake']
        self.loss_name = ['loss_G', 'loss_D']
        self.model_name = ['G', 'D']

        self.noise_dim = 64
        self.fixed_noise = torch.randn((opt.batch_size, self.noise_dim)).to(opt.device)
        self.G = Generator(self.noise_dim, self.img_size).to(opt.device)
        self.D = Discriminator(self.img_size).to(opt.device)
        self.adversarial_loss = nn.BCELoss()

        self.get_net = self.get_network(self.model_name, net=(self.G, self.D))

        self.optimize_D = optim.Adam(self.D.parameters(), lr=opt.lr)
        self.optimize_G = optim.Adam(self.G.parameters(), lr=opt.lr)

    def set_input(self, input, label):
        self.real = input.view(input.size(0),-1).to(self.opt.device)
        self.noise = torch.randn((input.size(0), self.noise_dim)).to(self.opt.device)

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
            real = self.real.reshape(-1, self.opt.c, self.opt.resize, self.opt.resize)
            fake = self.G(self.fixed_noise).reshape(-1, self.opt.c, self.opt.resize, self.opt.resize)
            return [real[0:25], fake[0:25]]
