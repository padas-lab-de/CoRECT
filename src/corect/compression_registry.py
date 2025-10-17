from typing import Dict

from corect.quantization.AbstractCompression import AbstractCompression
from corect.quantization.BinaryCompression import THRESHOLD_TYPES, BinaryCompression
from corect.quantization.FloatCompression import PRECISION_TYPE, FloatCompression
from corect.quantization.LSHCompression import NUM_LSH_BITS, LSHCompression
from corect.quantization.MinMaxCompression import MinMaxCompression
from corect.quantization.PercentileCompression import NUM_BITS, PercentileCompression
from corect.quantization.PrincipalComponentAnalysis import OUT_DIMS as PCA_OUT_DIMS, PrincipalComponentAnalysis
from corect.quantization.ProductQuantization import COMBINATIONS, ProductQuantization


class CompressionRegistry:
    """
    A registry for compression methods used in the CoRECT project.
    This class is used to register and access different compression methods.
    """

    # Dictionary containing all compression methods to be used by the evaluate script
    _compression_methods: Dict[str, AbstractCompression] = {}

    @classmethod
    def get_compression_methods(cls) -> Dict[str, AbstractCompression]:
        """
        Returns the dictionary containing all registered compression methods.
        """
        return cls._compression_methods

    @classmethod
    def clear(cls):
        """
        Clears the compression_methods dictionary.
        This is useful for resetting the registry before adding new compression methods.
        """
        cls._compression_methods.clear()

    @classmethod
    def add_baseline(cls):
        """
        Adds the baseline compression method to the compression_methods dictionary.
        The baseline is a FloatCompression with 32-bit precision.
        """
        cls._compression_methods["32_full"] = FloatCompression("full")

    @classmethod
    def add_compressions(cls):
        """
        Adds all compression methods to the compression_methods dictionary.
        This includes different float precisions, percentile and equal distance compressions,
        and binary compressions with various thresholds.
        """
        for precision, num_bits in PRECISION_TYPE.items():
            cls._compression_methods[f"{num_bits}_{precision}"] = FloatCompression(
                precision
            )
        for bits in NUM_BITS:
            cls._compression_methods[f"{bits}_percentile"] = PercentileCompression(bits)
            cls._compression_methods[f"{bits}_equal_distance"] = MinMaxCompression(
                bits, 2.5
            )
        for threshold in THRESHOLD_TYPES:
            cls._compression_methods[f"1_binary_{threshold}"] = BinaryCompression(
                threshold
            )
        for name, (num_subvectors, code_size) in COMBINATIONS.items():
            cls._compression_methods[name] = ProductQuantization(
                num_subvectors, code_size
            )
        for pca_name, out_dim in PCA_OUT_DIMS.items():
            cls._compression_methods[pca_name] = PrincipalComponentAnalysis(
                n_components=out_dim
            )
        for bits in NUM_LSH_BITS:
            cls._compression_methods[f"{bits}_lsh"] = LSHCompression(bits)
