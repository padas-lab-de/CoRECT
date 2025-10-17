from typing import Tuple

import numpy as np
import torch

from corect.quantization.AbstractCompression import AbstractCompression


NUM_BITS = [2, 4, 8]


class MinMaxCompression(AbstractCompression):
    """
    Class for quantizing embedding vectors using the minimum and maximum values per embedding dimension.
    """

    def __init__(self, num_bits: int = 8, clip_percentile: float = 2.5):
        """
        Initializes the class with the number of bits and the percentiles to clip.

        Args:
            num_bits: The number of bits the resulting embeddings should occupy.
            clip_percentile: The percentile of outliers that should be clipped before quantization.
        """
        assert 1 < num_bits < 16
        assert 0 <= clip_percentile < 50
        self.num_bits = num_bits
        self.clip_percentile = clip_percentile
        self.mins = []
        self.maxs = []
        self.batch_mins = None
        self.batch_maxs = None

    def _compute_batch_min_max(self, embeddings: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Computes the batch-wide minimum and maximum per embedding dimension or uses pre-computed values to compress
        queries.

        Args:
            embeddings: The embedding tensor used to compute the minimum and maximum values.
        Returns:
            The minimum and maximum values per dimension.
        """
        if self.batch_mins is not None and self.batch_maxs is not None:
            mins, maxs = self.batch_mins, self.batch_maxs
            self.batch_mins, self.batch_maxs = None, None
            return mins, maxs
        else:
            out_embeds = embeddings
            if self.clip_percentile > 0:
                min_perc, max_perc = self.clip_percentile, 100 - self.clip_percentile
                points = np.percentile(embeddings.cpu().numpy(), [min_perc, max_perc], axis=0)
                out_embeds = torch.clip(embeddings.cuda(), torch.tensor(points[0]).cuda(),
                                        torch.tensor(points[1]).cuda())
            self.batch_mins = torch.min(out_embeds, dim=0).values.cuda()
            self.batch_maxs = torch.max(out_embeds, dim=0).values.cuda()
            return self.batch_mins, self.batch_maxs

    def compute_threshold(self, embeddings: torch.tensor):
        """
        Computes the minimum and maximum values per embedding dimension after clipping them. The computed values are
        later used during quantization.
        """
        if self.clip_percentile > 0:
            min_perc, max_perc = self.clip_percentile, 100 - self.clip_percentile
            points = np.percentile(embeddings.numpy(), [min_perc, max_perc], axis=0)
            embeddings = torch.clip(embeddings, torch.tensor(points[0]), torch.tensor(points[1]))
        self.mins = torch.min(embeddings, dim=0).values.cuda()
        self.maxs = torch.max(embeddings, dim=0).values.cuda()

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantizes the embeddings by first clipping a fixed percentile of minimum and maximum values per embedding
        dimension (optional) before using the minimum and maximum value per dimension to calculate the bin boundaries
        using 2**num_bits - 1 steps of equal length from minimum to maximum. The embedding values are then converted to
        their respective bin numbers.

        Args:
            embeddings: The embeddings to quantize.

        Returns:
            The quantized embeddings.
        """
        mins = self.mins[:len(embeddings[0])] if len(self.mins) >= len(embeddings[0]) else self.mins
        maxs = self.maxs[:len(embeddings[0])] if len(self.maxs) >= len(embeddings[0]) else self.maxs
        out_embeds = embeddings.cuda()

        if len(mins) == 0 or len(maxs) == 0:
            mins, maxs = self._compute_batch_min_max(embeddings)

        steps = (maxs - mins) / (2**self.num_bits - 1)
        return torch.floor((out_embeds - mins) / steps) - int(2**self.num_bits * 0.5)
