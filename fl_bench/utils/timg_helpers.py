import json
import os
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data.dataset import Dataset
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from dotenv import load_dotenv

load_dotenv()


def _get_site_class_summary(train_label, site_idx):
    class_sum = {}

    for site, data_idx in site_idx.items():
        unq, unq_cnt = np.unique(train_label[data_idx], return_counts=True)
        tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        class_sum[site] = tmp
    return class_sum


class timgDataSplitter(FLComponent):
    def __init__(
        self,
        split_dir: str = None,
        num_sites: int = 10,
        alpha: float = 0.5,
        seed: int = 0,
    ):
        super().__init__()
        self.split_dir = split_dir
        self.num_sites = num_sites
        self.alpha = alpha
        self.seed = seed

        if alpha < 0.0:
            raise ValueError(f"Alpha should be non-negative but was {alpha}.")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.split(fl_ctx)

    def split(self, fl_ctx: FLContext):
        np.random.seed(self.seed)

        self.log_info(
            fl_ctx,
            f"Partition timg dataset into {self.num_sites} sites with alpha {self.alpha}",
        )
        site_idx, class_sum = self._partition_data()

        # write to files
        if self.split_dir is None:
            raise ValueError(
                "You need to define a valid `split_dir` when splitting the data."
            )
        if not os.path.isdir(self.split_dir):
            os.makedirs(self.split_dir)
        sum_file_name = os.path.join(self.split_dir, "summary.txt")
        with open(sum_file_name, "w") as sum_file:
            sum_file.write(f"Number of clients: {self.num_sites} \n")
            sum_file.write(f"Dirichlet sampling parameter: {self.alpha} \n")
            sum_file.write("Class counts for each client: \n")
            sum_file.write(json.dumps(class_sum))

        site_file_path = os.path.join(self.split_dir, "site-")
        for site in range(self.num_sites):
            site_file_name = site_file_path + str(site + 1) + ".npy"
            np.save(site_file_name, np.array(site_idx[site]))

    def load_timg_data(self):
        # download data
        train_dataset = datasets.ImageFolder(os.path.join(os.getenv("TIMG_ROOT"), "/train"))
        # only training label is needed for doing split
        train_label = np.array(train_dataset.targets)
        return train_label

    def _partition_data(self):
        train_label = self.load_timg_data()

        min_size = 0
        K = 200
        N = train_label.shape[0]
        site_idx = {}

        # split
        while min_size < 200:
            idx_batch = [[] for _ in range(self.num_sites)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(train_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_sites))
                # Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / self.num_sites)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        # shuffle
        for j in range(self.num_sites):
            np.random.shuffle(idx_batch[j])
            site_idx[j] = idx_batch[j]

        # collect class summary
        class_sum = _get_site_class_summary(train_label, site_idx)

        return site_idx, class_sum


class Timg_Idx(Dataset):
    def __init__(self, root, data_idx=None, train=True, transform=None):
        """Timg dataset with index to extract subset

        Args:
            root: data root
            data_idx: to specify the data for a particular client site.
                If index provided, extract subset, otherwise use the whole set
            train: whether to use the training or validation split (default: True)
            transform: image transforms
            download: whether to download the data (default: False)
        Returns:
            A PyTorch dataset
        """
        self.root = root
        self.data_idx = data_idx
        self.train = train
        self.transform = transform
        self.data, self.target = self.__build_timg_subset__()

    def __build_timg_subset__(self):
        # if index provided, extract subset, otherwise use the whole set

        if self.train:
            timg_dataobj = datasets.ImageFolder(
                os.path.join(os.getenv("TIMG_ROOT"), "/train"),
                self.transform
            )
        else:
            timg_dataobj = datasets.ImageFolder(
                os.path.join(os.getenv("TIMG_ROOT"), "/valid"),
                self.transform
            )
        data = timg_dataobj.data
        target = np.array(timg_dataobj.targets)
        if self.data_idx is not None:
            data = data[self.data_idx]
            target = target[self.data_idx]
        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


def main():
    splitter = timgDataSplitter()
    splitter.load_timg_data()


if __name__ == "__main__":
    main()
