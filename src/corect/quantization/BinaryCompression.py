import torch

from corect.quantization.AbstractCompression import AbstractCompression


THRESHOLD_TYPES = ["zero", "median"]


class BinaryCompression(AbstractCompression):
    """
    Class for binary compression using a threshold to binarize embedding vectors.
    """

    def __init__(self, threshold_type: str = "zero"):
        """
        Initializes the class with the type of threshold to use.

        Args:
            threshold_type: The type of threshold to use, i.e. the median per embedding dimension or zero.
        """
        assert threshold_type in THRESHOLD_TYPES
        self.threshold_type = threshold_type
        self.median = []
        self.batch_median = None

    def compute_threshold(self, embeddings: torch.tensor):
        """
        Computes the median threshold per dimension over all passed embeddings. This threshold is then used in compress
        function.

        :param embeddings: Embedding vectors over which to compute the median.
        """
        if self.threshold_type == "median":
            self.median = torch.median(embeddings, dim=0).values

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Binarizes the given embeddings vectors using a zero threshold or the median per embedding dimension, depending
        on the defined threshold type.

        Args:
            embeddings: The embeddings to compress.

        Returns:
            The binarized embedding vectors.
        """
        if self.threshold_type == "zero":
            return torch.where(embeddings > 0, 1, 0).float()
        elif len(self.median) >= len(embeddings[0]):
            return torch.where(embeddings > self.median[:len(embeddings[0])], 1, 0).float()
        else:
            if self.batch_median is not None:
                embeds = torch.where(embeddings > self.batch_median[:len(embeddings[0])], 1, 0).float()
                self.batch_median = None
                return embeds
            else:
                self.batch_median = torch.median(embeddings, dim=0).values
                return torch.where(embeddings > self.batch_median[:len(embeddings[0])], 1, 0).float()
