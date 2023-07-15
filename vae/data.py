from typing import Callable, Dict, Tuple, Union

import torch
from sklearn.model_selection import train_test_split as tts
from torch.utils.data import Dataset as torchDataset


class Dataset(torchDataset):
    """
    Dataset class to be used for preprocessing and feeding ratings data to the VAE model
    """

    def __init__(
        self,
        cases: Dict = None,
        n_features: int = None,
        rating_function: Callable = lambda x: x,
    ):
        self.rf = rating_function
        self.n_features = n_features
        self.cases = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        indicies = list(map(lambda x: x[0], self.cases[index]))
        values = list(map(lambda x: self.rf(x[1]), self.cases[index]))

        tensor = torch.sparse_coo_tensor(
            torch.tensor(indicies).unsqueeze(0),
            torch.tensor(values, dtype=torch.float32),
            torch.Size([self.n_features]),
        ).to_dense()

        return tensor
