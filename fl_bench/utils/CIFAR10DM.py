import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fl_bench.utils.cifar10_helpers import CIFAR10_Idx


class CIFAR10DataModule():
    def __init__(
        self,
        data_dir: str,
        train_idx_root: str,
        client_idx: str,
        central: bool,
        batch_size: int = 64
    ):
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
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )
        self.transform_valid = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
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
            root=self.data_dir,
            train=False,
            download=False,
            transform=self.transform_valid
            )

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
