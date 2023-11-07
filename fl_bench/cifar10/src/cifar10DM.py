import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from fl_bench.cifar10.src.utils import CIFAR10_Idx, seed_everything


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CIFAR10DataModule:
    def __init__(
        self, data_dir: str, train_idx_root: str, client_idx: str, central: bool, batch_size: int = 64, seed: int = 0
    ):
        self.seed = seed
        seed_everything(seed)
        self.data_dir = data_dir
        self.client_idx = client_idx
        self.train_idx_root = train_idx_root
        self.central = central
        self.batch_size = batch_size
        self.transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                ),
                transforms.Resize((224, 224), antialias=True),
            ]
        )
        self.transform_valid = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
                transforms.Resize((224, 224), antialias=True),
            ]
        )
        self.setup()

    def setup(self):
        if self.central is False:
            site_idx_file_name = os.path.join(self.train_idx_root, self.client_idx + ".npy")
            if os.path.exists(site_idx_file_name):
                site_idx = np.load(site_idx_file_name).tolist()
            else:
                raise ValueError(f"File {site_idx_file_name} does not exist!")
        else:
            site_idx = None

        self.train_dataset = CIFAR10_Idx(
            root=self.data_dir,
            data_idx=site_idx,
            train=True,
            download=False,
            transform=self.transform_train,
        )
        self.val_dataset = datasets.CIFAR10(
            root=self.data_dir, train=False, download=False, transform=self.transform_valid
        )

        g = torch.Generator()
        g.manual_seed(self.seed)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

    def teardown(self):
        self.train_dataset = {}
        self.val_dataset = {}
        self.train_dataloader = {}
        self.val_dataloader = {}
