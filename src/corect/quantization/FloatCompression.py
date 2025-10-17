import torch

from corect.quantization.AbstractCompression import AbstractCompression


PRECISION_TYPE = {
    "full": 32,
    "half": 16,
    "bfloat16": 16,
    "float8m3": 8,
    "float8m2": 8
}


class FloatCompression(AbstractCompression):
    """
    Class for casting embedding vectors to float16 or leave them as full-precision vectors.
    """

    def __init__(self, precision_type: str = "half"):
        """
        Initializes the class with the precision type to use, i.e. full or half.

        Args:
            precision_type: The precision type.
        """
        assert precision_type in PRECISION_TYPE
        self.precision_type = precision_type

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Converts embeddings to float16, if the precision type is half, otherwise returns the full-precision vectors.

        Args:
            embeddings: The embeddings to convert.

        Returns:
            The converted embeddings.
        """
        if self.precision_type == "full":
            return embeddings
        elif self.precision_type == "half":
            return embeddings.type(torch.float16)
        elif self.precision_type == "bfloat16":
            return embeddings.type(torch.bfloat16)
        elif self.precision_type == "float8m3":
            # Cast to float8 with 3-bit mantissa, then back to float16 for similarity calculation
            return embeddings.type(torch.float8_e4m3fn).type(torch.float16)
        elif self.precision_type == "float8m2":
            # Cast to float8 with 2-bit mantissa, then back to float16 for similarity calculation
            return embeddings.type(torch.float8_e5m2).type(torch.float16)
        else:
            raise NotImplementedError(
                f"Cannot convert embedding to invalid precision type {self.precision_type}!"
            )
