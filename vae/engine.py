from typing import List

import torch

from core.data import Dataset
from core.engine import Engine as coreEngine
from vae.model import Model


class Engine(coreEngine):
    def __init__(self, model: Model, dataset: Dataset):
        self.model = model
        self.dataset = dataset
        self.raw2inner = dataset.trainset.to_inner_iid
        self.inner2raw = dataset.trainset.to_raw_iid

    def recommend(self, preferences: List[int | str]) -> List:
        """
        Method that takes user preferences and returns a list of recommendations based on said preferences.

        Parameters
        ----------
        preferences : List[int]
            List of raw item ids that the user preferes

        Returns
        -------
        List
            List of raw item ids that the system recommends sorted by ranking criteria

        Raises
        ------
        TypeError
            Raised if the preference parameter isn't a list of raw item ids
        """
        if not isinstance(preferences, List[int | str]):
            raise TypeError("preferences must be a list of raw item ids")

        # convert preferences to required format
        tensor = self.prefs_to_model_input(preferences)

        # feed preferences into the model and get the output
        output = self.model.forward(tensor)

        # perform necessary conversions
        indices = output.nonzero()
        idx_val = [(tuple(idx), output[idx].item()) for idx in indices]

        # sort by ranking criteria
        idx_val = sorted(idx_val, key=lambda x: x[1], reverse=True)

        # convert to raw ids and return
        recommended = list(map(lambda x: x[0], idx_val))
        recommended = list(map(self.inner2raw, recommended))

        return recommended

    def prefs_to_model_input(self, preferences: List[int | str]) -> torch.Tensor:
        """Method for converting user preferences to model input

        Parameters
        ----------
        preferences : List[int  |  str]
            raw user preferences

        Returns
        -------
        torch.Tensor
            tensor representing VAE input
        """
        inner_ids = list(map(self.raw2inner, preferences))
        indices = torch.tensor(inner_ids).unsqueeze(0)
        tensor = torch.sparse_coo_tensor(
            indices=indices,
            values=torch.ones(len(indices), dtype=torch.float32),
            size=torch.Size([self.dataset.n_features]),
        ).to_dense()

        return tensor
