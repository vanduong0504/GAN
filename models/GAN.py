import torch
from utils import *
import torch.nn as nn
import torch.optim as optim
from models.base_model import base


class Generator(nn.Module):
    def __init__(self, noise_dim, img_size):
        super(Generator, self).__init__()

        self.layer1 = self.make_layer(in_cha=noise_dim, out_cha=128)
        self.layer2 = self.make_layer(in_cha=128, out_cha=256)
        self.layer3 = self.make_layer(in_cha=256, out_cha=512)
        self.layer4 = self.make_layer(in_cha=512, out_cha=256)
        self.layer5 = self.make_layer(in_cha=256, out_cha=128)
        self.layer6 = self.make_layer(in_cha=128, out_cha=img_size, type=False)
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)

        return self.tanh(output)

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


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()

        self.layer1 = self.make_layer(in_cha=img_size, out_cha=512)
        self.layer2 = self.make_layer(in_cha=512, out_cha=256)
        self.layer3 = self.make_layer(in_cha=256, out_cha=1, type=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)

        return self.sigmoid(output)

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
            layer += [nn.Dropout(p=0.5)]

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

        self.loss_D = (self.loss_real + self.loss_fake) * 0.5
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """
        This function use to build calculate the loss of Generator.
        """
        pred_fake = self.D(self.fake)
        self.loss_G = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake))
        self.loss_G.backward()

    def optimize_parameters(self):
        """
        This function combine of Genrator loss, Discriminator loss and optimizer step for one epoch.
        """
        self.forward()

        # Discriminator
        self.optimize_D.zero_grad()
        self.backward_D()
        self.optimize_D.step()

        # Genrator
        self.optimize_G.zero_grad()
        self.backward_G()
        self.optimize_G.step()

        return [self.loss_G.item(), self.loss_D.item()]

    def evaluate_model(self):
        with torch.no_grad():
          real = self.real.reshape(-1, self.opt.c, self.opt.resize, self.opt.resize)
          fake = self.G(self.fixed_noise).reshape(-1, self.opt.c, self.opt.resize, self.opt.resize)
          return [real[0:25], fake[0:25]]
