import torch
from faiss import ProductQuantizer
from corect.quantization.AbstractCompression import AbstractCompression


COMBINATIONS = {
    "1024_2_pq": (1024, 2),
    "1024_8_pq": (1024, 8),
    "512_2_pq": (512, 2),
    "512_8_pq": (512, 8),
    "128_2_pq": (128, 2),
    "128_8_pq": (128, 8),
}


class ProductQuantization(AbstractCompression):
    """
    Implements Product Quantization for compressing embedding vectors.
    """

    def __init__(self, num_subvectors: int = 1024, code_size: int = 8):
        """
        Initializes the Product Quantization compressor.

        Args:
            num_subvectors: Number of sub-vectors to split the embedding into.
            code_size: Number of bits used to represent each sub-vector space.
        """
        self.num_subvectors = num_subvectors
        self.code_size = code_size
        self.pq = None
        self.batch_pq = None

    def fit(self, embeddings: torch.Tensor):
        """
        Learns the PQ codebooks from the given embeddings.

        Args:
            embeddings: Embeddings to fit PQ on. Shape: [n_samples, embedding_dim]
        """
        if self.pq is None:
            self.pq = ProductQuantizer(embeddings.shape[1], self.num_subvectors, self.code_size)
            self.pq.train_type = self.pq.Train_shared
        self.pq.train(embeddings.detach().cpu())

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compresses embeddings using the learned codebooks.

        Args:
            embeddings: Embeddings to compress. Shape: [n_samples, embedding_dim]

        Returns:
            A tensor of shape [n_samples, embedding_dim] that has been compressed using PQ, then decompressed for
            retrieval.
        """
        embeddings = embeddings.detach().cpu()
        if self.pq is None and self.batch_pq is None:
            self.batch_pq = ProductQuantizer(embeddings.shape[1], self.num_subvectors, self.code_size)
            self.batch_pq.train_type = self.batch_pq.Train_shared
            self.batch_pq.train(embeddings)
            return self.decompress(self.batch_pq.compute_codes(embeddings))
        elif self.batch_pq is not None:
            embeds = self.decompress(self.batch_pq.compute_codes(embeddings))
            self.batch_pq = None
            return embeds

        return self.decompress(self.pq.compute_codes(embeddings))

    def decompress(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decompresses the quantized codes into approximate embeddings.

        Args:
            codes: Tensor of shape [n_samples, num_subvectors] containing centroid indices.

        Returns:
            Decompressed embeddings (approximate) of shape [n_samples, embedding_dim].
        """
        assert self.pq is not None or self.batch_pq is not None, "ProductQuantization must be fit before decompressing."
        if self.batch_pq is not None:
            return torch.tensor(self.batch_pq.decode(codes))
        return torch.tensor(self.pq.decode(codes))
