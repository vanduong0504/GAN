from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST


class DATASET:
    def __init__(self, dataset, save_folder, resize):
        self.save_folder = save_folder
        self.resize = resize
        if dataset == "mnist":
            self.train = MNIST(root=self.save_folder, train=True, download=True, transform=self.transform())

    def transform(self):
        return transforms.Compose(
            [transforms.Resize(self.resize),
             transforms.ToTensor(),
             transforms.Normalize(0.5, 0.5)])

    def loader(self, dataset, batch_sizes):
        return DataLoader(dataset=dataset, batch_size=batch_sizes, shuffle=True, num_workers=2)
