import os
import time
import torch
from utils import *
import numpy as np
from tqdm import tqdm
from dataset import DATASET
from option import Options
from models import GAN, DCGAN, WGAN, cGAN


class net:
    def __init__(self):
        opt = Options().parse()
        self.opt = opt

        self.path_log = os.path.join(opt.log_dir, opt.model, opt.dataset)
        self.image_dir = os.path.join(opt.image_dir, opt.model, opt.dataset, opt.phase)

    def build_model(self):
        """
        This function to build the dataset and model from class Option.
        """
        data = DATASET(self.opt.dataset, self.opt.dataroot, self.opt.resize)
        self.loader = data.loader(data.train, self.opt.batch_size)

        if self.opt.model == "GAN":
            self.net = GAN.Model(self.opt)
        elif self.opt.model == "DCGAN":
            self.net = DCGAN.Model(self.opt)
        elif self.opt.model == "WGAN":
            self.net = WGAN.Model(self.opt)
        elif self.opt.model == "cGAN":
            self.net = cGAN.Model(self.opt)


        if self.opt.phase == "train":
            if self.opt.base_epoch: # Resume training
                self.net.load_networks(self.opt.base_epoch)
            else:
                for name, model in self.net.get_net.items():
                    init_weight(model, name)

    def train(self):
        opt = self.opt
        net = self.net
        loader = self.loader

        write_loss = create_logger(self.path_log, "Loss", self.net.loss_name)
        write_grad = create_logger(self.path_log, "Grad", self.net.model_name)
        write_image = create_logger(self.path_log, "Image", self.net.image_name)

        for epoch in range(opt.base_epoch + opt.epoch):
            loss = current_losses(net.loss_name)
            loop = tqdm(enumerate(loader), total=len(loader), position=0, leave=True)
            for batch_idx, (inputs, labels) in loop:
                
                net.train()
                net.set_input(inputs, labels)
                batch_loss = net.optimize_parameters()
                
                # Training display
                loop.set_description(f"Epoch [{epoch+1}/{opt.epoch}]")
                loop.set_postfix(OrderedDict(zip(net.loss_name, batch_loss)))
                time.sleep(0.1)

                for i, keys in enumerate(loss):
                    loss[keys].append(batch_loss[i])
                
                # Tensorboard image
                if batch_idx == 0:
                    net.eval()
                    image = grid_image(net.evaluate_model())
                    save_result(image[-1], self.image_dir, epoch+1) # Take generated images
                    for i, (name, writer) in enumerate(write_image.items()):
                        writer.add_image(name, image[i], global_step=epoch)

                # Tensorboard gradient: debug purpose
                for name, writer in write_grad.items():
                  sub_model = getattr(self.net, name)
                  for layer, param in sub_model.named_parameters():
                    writer.add_histogram(layer, param.grad, batch_idx)

            # Tensorboard loss
            for name, writer in write_loss.items():
                writer.add_scalar(name, np.mean(loss[keys]), global_step=epoch)

            if epoch % opt.save_freq == opt.save_freq - 1:
                print("SAVE")
                net.save_networks(epoch + 1)

            print(f"Epoch[{epoch+1}/{opt.epoch}]: [" +
            ", ".join([f"{key}={np.mean(list(filter(lambda num: num != 0, value))):.4f}" 
            for key, value in loss.items()]) + "]")
            
    def test(self):
        self.build_model()
        self.net.load_networks(self.opt.epoch)
        self.net.G.eval()

        dump_tensor = torch.randn(self.opt.batch_size, 1) # Create a dump tensor because set_input() need batchsize
        if self.opt.model in ["GAN", "WGAN", "DCGAN"]:
            self.net.set_input(dump_tensor)
        elif self.opt.model == "cGAN":
            label = torch.randint(0, self.opt.num_class, size=self.opt.bach_size) # Generate random labels
            self.net.set_input(dump_tensor, label)

        with torch.no_grad():
            self.net.forward()
            if self.opt.model == "GAN":
                self.net.fake = self.net.fake.reshape(-1, self.opt.c, self.opt.resize, self.opt.resize)
            save_result(self.net.fake, self.image_dir)
