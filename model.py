import os
import time
from utils import *
import numpy as np
from tqdm import tqdm
from dataset import DATASET
from option import Options
from models import GAN


class net:
    def __init__(self):
        opt = Options().parse()
        self.opt = opt

    def build_model(self):
        """
        This function to build the dataset and model from class Option.
        """
        data = DATASET(dataset=self.opt.dataset, save_folder=self.opt.dataroot, resize=self.opt.resize)
        self.loader = data.loader(data.train, batch_sizes=self.opt.batch_size)

        if self.opt.model == "GAN":
            self.net = GAN.GANModel(self.opt)

        if self.opt.base_epoch:
            self.net.load_networks(self.opt.base_epoch)
        else:
            for name, model in self.net.get_net.items():
                init_weight(model, name)
            print("Finishing!!!", end='\n\n')

    def train(self):
        opt = self.opt
        net = self.net
        loader = self.loader

        path_log = os.path.join(opt.log_dir, opt.model, opt.dataset)
        write_loss = create_logger(path_log, "Loss", self.net.loss_name)
        write_image = create_logger(path_log, "Image", self.net.image_name)

        for epoch in range(opt.base_epoch + opt.epoch):
            net.train()
            # Use for print loss while training
            loss = current_losses(net.loss_name)
            loop = tqdm(enumerate(loader), total=len(loader), position=0, leave=True)
            for batch_idx, (inputs, labels) in loop:

                net.set_input(inputs, labels)
                net.forward()
                batch_loss = net.optimize_parameters()

                loop.set_description(f"Epoch [{epoch+1}/{opt.epoch}]")
                loop.set_postfix(OrderedDict(zip(net.loss_name, batch_loss)))
                time.sleep(0.1)

                for i, keys in enumerate(loss):
                    loss[keys].append(batch_loss[i])

                if (batch_idx + 1) % len(loader) == 0:
                    image = grid_image(net.image_name, net.evaluate_model())
                    for i, (keys, writer) in enumerate(write_image.items()):
                        writer.add_image(keys, image[i], global_step=epoch)

            for keys, writer in write_loss.items():
                writer.add_scalar(keys, np.mean(loss[keys]), global_step=epoch)

            if epoch % opt.save_freq == opt.save_freq - 1:
                print('SAVE')
                net.save_networks(epoch + 1)

            print(f'Epoch[{epoch+1}/{opt.epoch}]: ' + ' '.join(f'{key}: {np.mean(value)}' for key, value in loss.items()))

    def test(self):
        if self.opt.model in ("GAN", "cGAN", "DCGAN"):
            self.net.load_networks()
