from abc import ABC, abstractmethod

import torch


class AbstractCompression(ABC):
    """
    Base class for compression methods.
    """

    @abstractmethod
    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compress the embeddings and return them.
        """
        pass
