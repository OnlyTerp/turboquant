"""
TurboQuant — Near-Optimal KV Cache Compression for LLM Inference
=================================================================

First open-source implementation of TurboQuant (ICLR 2026).
Compress your LLM's KV cache by 5× with near-zero quality loss.

Algorithm: Random rotation → Scalar Lloyd-Max quantization → QJL residual
Result: 3.5 bits/value = identical quality to FP16. Provably near-optimal.

Reference: https://arxiv.org/abs/2504.19874
Authors: Zandieh, Daliri, Hadian, Mirrokni (Google Research / NYU / Google DeepMind)
"""

__version__ = "0.2.0"
__author__ = "Terp AI Labs"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# Public API — Core (always available)
# ---------------------------------------------------------------------------
from .cache import (
    TurboQuantCache,
    TurboQuantConfig,
    TurboQuantCompressed,
    TurboQuantMSECompressed,
    PolarQuantCompressed,
    QJLCompressed,
    Codebook,
    RandomHadamardRotation,
    RandomOrthogonalRotation,
    MixedPrecisionConfig,
    # Core functions
    polarquant_encode,
    polarquant_decode,
    turboquant_mse_encode,
    turboquant_mse_decode,
    qjl_encode,
    turboquant_encode_internal,
    turboquant_decode_single,
    compute_lloyd_max_codebook,
    compute_online_codebook,
    generate_qjl_matrix,
    detect_outlier_channels,
    # Utilities
    compression_ratio_fp16,
    memory_bytes_per_vector,
    fwht,
    fwht_inplace,
)

# ---------------------------------------------------------------------------
# Optional: Triton GPU kernels (requires triton)
# ---------------------------------------------------------------------------
try:
    from .kernels import (
        fwht_kernel,
        polarquant_encode_kernel,
        polarquant_decode_kernel,
        qjl_encode_kernel,
        turboquant_attention_kernel,
        torch_fwht,
        torch_polarquant_encode,
        torch_polarquant_decode,
        torch_qjl_encode,
        torch_turboquant_attention,
    )
    _HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    _HAS_TRITON = False

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    # Core
    "TurboQuantCache",
    "TurboQuantConfig",
    "TurboQuantCompressed",
    "TurboQuantMSECompressed",
    "PolarQuantCompressed",
    "QJLCompressed",
    "Codebook",
    "RandomHadamardRotation",
    "RandomOrthogonalRotation",
    "MixedPrecisionConfig",
    "polarquant_encode",
    "polarquant_decode",
    "turboquant_mse_encode",
    "turboquant_mse_decode",
    "qjl_encode",
    "turboquant_encode_internal",
    "turboquant_decode_single",
    "compute_lloyd_max_codebook",
    "compute_online_codebook",
    "generate_qjl_matrix",
    "detect_outlier_channels",
    "compression_ratio_fp16",
    "memory_bytes_per_vector",
    "fwht",
    "fwht_inplace",
]
