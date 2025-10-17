import torch
from faiss import PCAMatrix

from corect.quantization.AbstractCompression import AbstractCompression


OUT_DIMS = {
    "512_pca": 512,
    "256_pca": 256,
    "128_pca": 128,
    "64_pca": 64,
    "32_pca": 32
}


class PrincipalComponentAnalysis(AbstractCompression):
    """
    Class for compressing embedding vectors using Principal Component Analysis using FAISS.
    """

    def __init__(self, n_components: int = 512):
        """
        Initializes the class with the dimensionality of the embedding vectors and the dimension to which they should be
        reduced.

        Args:
            num_dims (int): Number of dimensions of the embedding vectors.
            n_components (int): Number of dimensions to compress to.
        """
        self.n_components = n_components
        self.pca = None
        self.batch_pca = None

    def fit(self, embeddings: torch.tensor):
        """
        Train PCA on the given embedding vectors.

        Args:
            embeddings (torch.tensor): The embedding vectors used to calculate the PCA matrix.
        """
        assert embeddings.shape[1] > self.n_components
        if self.pca is None:
            self.pca = PCAMatrix(embeddings.shape[1], self.n_components)
        self.pca.train(embeddings.detach().cpu())

    def compress(self, embeddings: torch.tensor) -> torch.tensor:
        """
        Applies the PCA transformation to the embeddings and returns the compressed vectors.

        Args:
            embeddings (torch.tensor): The embedding vectors to compress.

        Returns:
            The compressed embeddings.
        """
        embeddings = embeddings.detach().cpu()
        if self.pca is None and self.batch_pca is None:
            self.batch_pca = PCAMatrix(embeddings.shape[1], self.n_components)
            self.batch_pca.train(embeddings)
            return torch.tensor(self.batch_pca.apply(embeddings))
        elif self.batch_pca is not None:
            compressed_embeds = self.batch_pca.apply(embeddings)
            self.batch_pca = None
            return torch.tensor(compressed_embeds)
        else:
            return torch.tensor(self.pca.apply(embeddings))
