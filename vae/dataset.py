from typing import Callable, Dict, Tuple, Union

import torch
from sklearn.model_selection import train_test_split as tts
from torch.utils.data import Dataset as torchDataset

from core.dataset import Dataset as coreDataset


class Dataset(torchDataset):
    """
    Dataset class to be used for preprocessing ratings data for the VAE model
    """

    def __init__(
        self,
        dataset: coreDataset | Dict = None,
        n_features: int = None,
        rating_function: Callable = lambda x: x,
    ):
        self.rf = rating_function
        self.n_features = n_features
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        indicies = list(map(lambda x: x[0], self.data[index]))
        values = list(map(lambda x: self.rating_function(x[1]), self.data[index]))

        tensor = torch.sparse_coo_tensor(
            torch.tensor(indicies).unsqueeze(0),
            torch.tensor(values, dtype=torch.float32),
            torch.Size([self.n_features]),
        ).to_dense()

        return tensor

    def tvt_datasets(
        self, mode: Union["user", "item"] = "user"
    ) -> Tuple[Dataset, Dataset, Dataset]:
        train, valid, test = self.data.train_valid_test_split()
        train = Dataset(train, self.n_features, self.rating_function)
        valid = Dataset(valid, self.n_features, self.rating_function)
        test = Dataset(test, self.n_features, self.rating_function)
        return train, valid, test
