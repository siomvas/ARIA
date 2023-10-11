import json
import os
import random
from typing import Dict
import pprint

import medmnist
import numpy as np
import torch
from dotenv import load_dotenv
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from torch.utils.data.dataset import Dataset

load_dotenv()


def load_weights(model: torch.nn.Module, weights: Dict[str, np.ndarray]) -> torch.nn.Module:
    local_var_dict = model.state_dict()
    model_keys = weights.keys()
    for var_name in local_var_dict:
        if var_name in model_keys:
            w = weights[var_name]
            try:
                local_var_dict[var_name] = torch.as_tensor(np.reshape(w, local_var_dict[var_name].shape))
            except Exception as e:
                raise ValueError(f"Convert weight from {var_name} failed with error {str(e)}")
    model.load_state_dict(local_var_dict)
    return model


def extract_weights(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    local_state_dict = model.state_dict()
    local_model_dict = {}
    for var_name in local_state_dict:
        try:
            local_model_dict[var_name] = local_state_dict[var_name].cpu().numpy()
        except Exception as e:
            raise ValueError(f"Convert weight from {var_name} failed with error: {str(e)}")
    return local_model_dict


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


def read_json(filename):
    assert os.path.isfile(filename), f"{filename} does not exist!"

    with open(filename, "r") as f:
        return json.load(f)


def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def map_func(dataset: str, learner: str) -> str:
    """generates the proper learner path

    Args:
        dataset (str): name of the dataset
        learner (str): name of the learner

    Returns:
        learner_path (str): the path of the learner
    """

    pascal = dataset.replace("_", " ").title().replace(" ", "") + "Learner"
    learner_path = f"fl_bench.{dataset}.src.{dataset}_{learner}_learner.{pascal}"

    return learner_path


def _get_site_class_summary(label, site_idx):
    class_sum = {}

    for site, data_idx in site_idx.items():
        unq, unq_cnt = np.unique(label[data_idx], return_counts=True)
        tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        class_sum[site] = tmp
    return class_sum


class MedMNISTDataSplitter(FLComponent):
    def __init__(
        self,
        split_dir: str = None,
        num_sites: int = 5,
        alpha: float = 0.5,
        seed: int = 0,
    ):
        super().__init__()
        self.split_dir = split_dir
        self.num_sites = num_sites
        self.alpha = alpha
        self.seed = seed

        seed_everything(seed)

        if alpha < 0.0:
            raise ValueError(f"Alpha should be non-negative but was {alpha}.")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.split(fl_ctx)

    def split(self, fl_ctx: FLContext):
        np.random.seed(self.seed)

        self.log_info(
            fl_ctx,
            f"Partition OrganAMNIST training set into {self.num_sites} sites with Dirichlet sampling under alpha {self.alpha}",
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

    def load_medmnist_data(self):
        # download data
        train_dataset = medmnist.OrganAMNIST(
            root=os.getenv("MEDMNIST_ROOT"),
            split="train",
            download=True
        )

        # only training label is needed for doing split
        train_label = np.array(train_dataset.labels)
        return train_label

    def _partition_data(self):
        train_label = self.load_medmnist_data()

        min_size = 0
        K = 11
        N = train_label.shape[0]
        site_idx = {}

        # split
        while min_size < 10:
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


class Combo_Splitter(FLComponent):
    def __init__(
        self,
        split_dir: str = None,
        num_sites: int = 5,
        alpha: float = 0.5,
        seed: int = 0,
    ):
        super().__init__()
        self.split_dir = split_dir
        self.num_sites = num_sites
        self.alpha = alpha
        self.seed = seed

        seed_everything(seed)

        if alpha < 0.0:
            raise ValueError(f"Alpha should be non-negative but was {alpha}.")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.split(fl_ctx)

    def split(self, fl_ctx: FLContext):
        np.random.seed(self.seed)

        self.log_info(
            fl_ctx,
            f"Partition OrganAMNIST training and val sets into {self.num_sites} sites with Dirichlet sampling under alpha {self.alpha}",
        )
        site_idx_train, class_sum_train, site_idx_val, class_sum_val = self._partition_data()

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
            sum_file.write("Train Class counts for each client: \n")
            # sum_file.write(json.dumps(class_sum))
            pretty_json_str = pprint.pformat(json.dumps(class_sum_train), compact=True, indent=2).replace("'",'"')
            sum_file.write(pretty_json_str)
            sum_file.write("\n")
            sum_file.write("Val Class counts for each client: \n")
            pretty_json_str = pprint.pformat(json.dumps(class_sum_val), compact=True, indent=2).replace("'",'"')
            sum_file.write(pretty_json_str)

        site_file_path = os.path.join(self.split_dir, "site-")
        for site in range(self.num_sites):
            site_file_name = site_file_path + str(site + 1) + "-train" + ".npy"
            np.save(site_file_name, np.array(site_idx_train[site]))
            site_file_name = site_file_path + str(site + 1) + "-val" + ".npy"
            np.save(site_file_name, np.array(site_idx_val[site]))

    def load_medmnist_data(self):
        # download data
        train_dataset = medmnist.OrganAMNIST(
            root=os.getenv("MEDMNIST_ROOT"),
            split="train",
            download=True
        )
        val_dataset = medmnist.OrganAMNIST(
            root=os.getenv("MEDMNIST_ROOT"),
            split="val",
            download=True
        )

        # only labels needed
        train_label = np.array(train_dataset.labels)
        val_label = np.array(val_dataset.labels)
        return train_label, val_label

    def _partition_data(self):
        train_label, val_label = self.load_medmnist_data()

        min_size = 0
        K = 11
        N = train_label.shape[0]
        site_idx_train = {}

        # split train
        while min_size < 10:
            idx_batch_train = [[] for _ in range(self.num_sites)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(train_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_sites))
                # Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / self.num_sites)
                        for p, idx_j in zip(proportions, idx_batch_train)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch_train = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch_train, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch_train])

        # split val
        min_size = 0
        N = val_label.shape[0]
        site_idx_val = {}

        while min_size < 10:
            idx_batch_val = [[] for _ in range(self.num_sites)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(val_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_sites))
                # Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / self.num_sites)
                        for p, idx_j in zip(proportions, idx_batch_val)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch_val = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch_val, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch_val])

        # shuffle
        for j in range(self.num_sites):
            np.random.shuffle(idx_batch_train[j])
            site_idx_train[j] = idx_batch_train[j]

            np.random.shuffle(idx_batch_val[j])
            site_idx_val[j] = idx_batch_val[j]

        # collect class summaries
        class_sum_train = _get_site_class_summary(train_label, site_idx_train)
        class_sum_val = _get_site_class_summary(val_label, site_idx_val)

        return site_idx_train, class_sum_train, site_idx_val, class_sum_val


class MedMNIST_Idx(Dataset):
    def __init__(self, root, data_idx=None, split="train", transform=None, target_transform=None, download=False):
        """MedMNIST dataset with index to extract subset

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
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_medmnist_subset__()

    def __build_medmnist_subset__(self):
        # if index provided, extract subset, otherwise use the whole set
        dataobj = medmnist.OrganAMNIST(
            root=self.root,
            split=self.split,
            transform=self.transform,
            target_transform=self.target_transform,
        )
        # FIX:
        data = dataobj.imgs
        target = np.array(dataobj.labels)
        if self.data_idx is not None:
            data = data[self.data_idx]
            target = target[self.data_idx]
        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label": target}

    def __len__(self):
        return len(self.data)


def main():
    splitter = MedMNISTDataSplitter()
    splitter.load_medmnist_data()


if __name__ == "__main__":
    main()
