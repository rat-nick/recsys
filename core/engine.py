from abc import ABC, abstractmethod
from typing import Dict, List


class Engine(ABC):
    """
    Abstract interface that all recommender engine objects must implement

    """

    @abstractmethod
    def recommend(self, preferences: List | Dict) -> List:
        """
        Abstract method that all recommender engine objects must implement

        Parameters
        ----------
        preferences : List | Dict
            List or dict of user preferences

        Returns
        -------
        List
            Sorted list of recommendations
        """
        pass
