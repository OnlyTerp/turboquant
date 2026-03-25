"""
TurboQuant — 3-bit KV Cache Compression
=========================================

Extreme KV cache compression combining PolarQuant (MSE-optimal scalar
quantization) with QJL (1-bit residual correction) for near-optimal
vector quantization with unbiased inner product estimation.

4.9× compression vs FP16. 3.25 bits per value. 0.90+ cosine similarity.

Reference: https://arxiv.org/abs/2504.19874 (ICLR 2026)
"""

__version__ = "0.1.0"
__author__ = "Terp AI Labs"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# Public API — Cache
# ---------------------------------------------------------------------------
from .cache import (
    TurboQuantCache,
    TurboQuantConfig,
    TurboQuantCompressed,
    PolarQuantCompressed,
    QJLCompressed,
    Codebook,
    RandomHadamardRotation,
    # Core encode/decode functions
    polarquant_encode,
    polarquant_decode,
    qjl_encode,
    turboquant_encode_internal,
    turboquant_decode_single,
    compute_lloyd_max_codebook,
    generate_qjl_matrix,
    # Utilities
    compression_ratio_fp16,
    memory_bytes_per_vector,
)

# ---------------------------------------------------------------------------
# Public API — Kernels
# ---------------------------------------------------------------------------
from .kernels import (
    # Triton kernels
    fwht_kernel,
    polarquant_encode_kernel,
    polarquant_decode_kernel,
    qjl_encode_kernel,
    turboquant_attention_kernel,
    # PyTorch fallbacks
    torch_fwht,
    torch_polarquant_encode,
    torch_polarquant_decode,
    torch_qjl_encode,
    torch_turboquant_attention,
    # Constants
    CODEBOOK_CENTROIDS_LIST,
    CODEBOOK_BOUNDARIES_LIST,
)

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__license__",
    # Cache
    "TurboQuantCache",
    "TurboQuantConfig",
    "TurboQuantCompressed",
    "PolarQuantCompressed",
    "QJLCompressed",
    "Codebook",
    "RandomHadamardRotation",
    # Encode/Decode
    "polarquant_encode",
    "polarquant_decode",
    "qjl_encode",
    "turboquant_encode_internal",
    "turboquant_decode_single",
    "compute_lloyd_max_codebook",
    "generate_qjl_matrix",
    # Utilities
    "compression_ratio_fp16",
    "memory_bytes_per_vector",
    # Triton kernels
    "fwht_kernel",
    "polarquant_encode_kernel",
    "polarquant_decode_kernel",
    "qjl_encode_kernel",
    "turboquant_attention_kernel",
    # PyTorch fallbacks
    "torch_fwht",
    "torch_polarquant_encode",
    "torch_polarquant_decode",
    "torch_qjl_encode",
    "torch_turboquant_attention",
    # Constants
    "CODEBOOK_CENTROIDS_LIST",
    "CODEBOOK_BOUNDARIES_LIST",
]
