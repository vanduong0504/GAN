import argparse
from utils import *


class Options:
    """
    This class provide some basic arguments.
    """
    def initialize(self, parser):
        parser.add_argument("--model", type=str, default="GAN",
                            help="[GAN,...]")
        parser.add_argument("--phase", type=str, default="train",
                            help="[train / test]")
        parser.add_argument("--dataroot", type=str, default="./data",
                            help="Path to datasets. (default: `./data`)")
        parser.add_argument("--dataset", type=str, default="mnist",
                            help="Dataset name [mnist]")
        parser.add_argument("--c", default=1, type=int, metavar="CHANNELS",
                            help="Number of image channels. (default: 1)")
        parser.add_argument("--classes", default=10, type=int,
                            help="Number of classes. (default: 10)")
        parser.add_argument('--resize', type=int, default=28,
                            help='scale images to this size')
        #train option
        parser.add_argument("--epoch", default=50, type=int,
                            help="Number of total epochs to run. (default: 50)")
        parser.add_argument("--batch_size", default=32, type=int, metavar="BS",
                            help="Input batch size. (default: 32)")
        parser.add_argument("--lr", type=float, default=3e-4,
                            help="Learning rate. (default:3e-4)")
        parser.add_argument("--device", type=str, default="cuda",
                            help="Set gpu mode: [cpu, cuda]")
        parser.add_argument("--base_epoch", type=int, default=0,
                            help="base epoch to resume training")                    
        parser.add_argument("--result_dir", type=str, default="./results", metavar="RD",
                            help="Directory to save the results. (default: `./results`)")
        parser.add_argument("--log_dir", type=str, default="./logs", metavar="RD",
                            help="Directory to save the log. (default: `./logs`)")
        parser.add_argument("--save_freq", type=int, default=10, metavar="SF",
                            help="Number of epochs to save the latest results. (default: 10)")
        parser.add_argument("--save_path", type=str, default="./weights", metavar="SP",
                            help="Directory to save weight. (default: `./weights`)")
        #restore option
        parser.add_argument("--restore_G_path", type=str, default=None, metavar="rG",
                            help='the path to restore the generator.')
        parser.add_argument("--restore_D_path", type=str, default=None, metavar="rD",
                             help='the path to restore the discriminator.')
        return parser

    def check_args(self, args):
        """
        This function to check for the arguments.
        """
        #--result_dir
        check_folder(os.path.join(args.result_dir, args.model, args.dataset))

        #--epoch
        try:
            assert args.epoch >= 1
        except BaseException:
            print("Number of epoch must be larger than or equal to one")

        #--batch_size
        try:
            assert args.batch_size >= 1
        except BaseException:
            print("batch size must be larger than or equal to one")
        return args

    def print_options(self, opt):
        print()
        print("##### Information #####")
        print("# model : ", opt.model)
        print("# dataset : ", opt.dataset)
        print("# channels : ", opt.c)
        print("# classes : ", opt.classes)
        print("# epoch : ", opt.epoch)
        print("# batch_size : ", opt.batch_size)
        print("# save_freq  : ", opt.save_freq)
        print()

    def gather_options(self):
        parser = argparse.ArgumentParser(
            description="Generate Image on MNIST and FashionMNIST")
        self.parser = self.initialize(parser)

        return self.check_args(self.parser.parse_args())

    def parse(self):
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt
