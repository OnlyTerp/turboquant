"""
TurboQuant vLLM Plugin — 3.5-bit KV Cache Compression

TurboQuant combines PolarQuant (MSE-optimal scalar quantization) with
QJL (1-bit Quantized Johnson-Lindenstrauss residual correction) to
achieve extreme KV cache compression with near-zero accuracy loss.

Reference: https://arxiv.org/abs/2504.19874
"""

__version__ = "0.1.0"
__author__ = "TurboQuant Contributors"
__license__ = "Apache-2.0"

from vllm_plugin.config import TurboQuantConfig
from vllm_plugin.attention import (
    TurboQuantAttentionBackend,
    TurboQuantAttentionImpl,
)

__all__ = [
    "__version__",
    "TurboQuantConfig",
    "TurboQuantAttentionBackend",
    "TurboQuantAttentionImpl",
]
