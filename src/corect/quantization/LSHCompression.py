import torch
from faiss import IndexLSH

from corect.quantization.AbstractCompression import AbstractCompression


NUM_LSH_BITS = [8192, 4096, 2048, 1024]


class LSHCompression(AbstractCompression):
    """
    Implements LSH-based compression using FAISS.
    """

    def __init__(self, num_bits: int = 1024):
        """
        Initializes the class with the number of bits used to store each vector.

        Args:
            num_bits (int): Number of bits used per vector. Defaults to 1024.
        """
        self.num_bits = num_bits
        self.lsh = None
        self.batch_lsh = None

    def fit(self, embeddings: torch.Tensor):
        """
        Learns the hash functions to map the given embeddings to binary codes.

        Args:
            embeddings (torch.Tensor): The embedding vectors to compress.
        """
        if self.lsh is None:
            self.lsh = IndexLSH(embeddings.shape[1], self.num_bits)
        self.lsh.train(embeddings.detach().cpu())

    def compress(self, embeddings: torch.tensor) -> torch.tensor:
        """
        Applies LSH compression to the given embeddings and returns the decoded results.

        Args:
            embeddings (torch.Tensor): The embedding vectors to compress.

        Returns:
            The decoded LSH compressed embeddings.
        """
        embeddings = embeddings.detach().cpu()
        if self.lsh is None and self.batch_lsh is None:
            self.batch_lsh = IndexLSH(embeddings.shape[1], self.num_bits)
            self.batch_lsh.train(embeddings)
            return torch.tensor(self.batch_lsh.sa_decode(self.batch_lsh.sa_encode(embeddings)))
        elif self.batch_lsh is not None:
            embeds = torch.tensor(self.batch_lsh.sa_decode(self.batch_lsh.sa_encode(embeddings)))
            self.batch_lsh = None
            return embeds
        else:
            return torch.tensor(self.lsh.sa_decode(self.lsh.sa_encode(embeddings)))
