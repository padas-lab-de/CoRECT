import numpy as np
import torch

from corect.quantization.AbstractCompression import AbstractCompression


NUM_BITS = [2, 4, 8]


class PercentileCompression(AbstractCompression):
    """
    Class for quantizing embedding vectors using bins containing the same number of points.
    """

    def __init__(self, num_bits: int = 8):
        """
        Initializes the class with the number of bits the embeddings should be quantized to.

        Args:
            num_bits: The number of bits the resulting embeddings should occupy.
        """
        assert 1 < num_bits < 16
        self.num_bits = num_bits
        self.bin_edges = []
        self.batch_edges = None

    def compute_threshold(self, embeddings: torch.tensor):
        """
        Computes the bin edges for quantizing the given embeddings to the specified number of bits. The computed
        boundaries are subsequently used in the compress function during quantization.
        """
        quantiles = np.linspace(0, 100, num=2 ** self.num_bits + 1)
        self.bin_edges = torch.tensor(np.percentile(embeddings.numpy(), quantiles, axis=0)).cuda()

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantizes the embeddings by calculating all their 2**num_bits percentiles per embedding dimension. The
        percentile values are then used as boundaries such that embedding values are converted to the respective bin
        number.

        Args:
            embeddings: The embeddings to quantize.

        Returns:
            The quantized embeddings.
        """
        bin_edges = self.bin_edges[:len(embeddings[0])] if len(self.bin_edges) >= len(embeddings[0]) else self.bin_edges

        if len(bin_edges) == 0:
            if self.batch_edges is not None:
                bin_edges = self.batch_edges
                self.batch_edges = None
            else:
                quantiles = np.linspace(0, 100, num=2**self.num_bits + 1)
                bin_edges = torch.tensor(np.percentile(embeddings.detach().cpu().numpy(), quantiles, axis=0)).cuda()
                self.batch_edges = bin_edges

        bin_indices = torch.empty_like(embeddings, dtype=torch.uint8).cuda()
        embeddings = embeddings.cuda()

        for col in range(embeddings.shape[1]):
            # Getting the corresponding bin indices from the values
            bin_indices[:, col] = torch.bucketize(
                embeddings[:, col], bin_edges[1:-1, col], right=False
            )

        return bin_indices.float()
