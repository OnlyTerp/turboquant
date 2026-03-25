"""
TurboQuant vLLM Plugin — Platform Plugin

Registers ``TurboQuantPlatform`` with vLLM's plugin system so that running
``vllm serve <model> --attention-backend turboquant`` routes all attention
through the TurboQuant-compressed KV cache backend.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# vLLM Platform import — graceful fallback
# ---------------------------------------------------------------------------
_VLLM_AVAILABLE = False
_Platform: Optional[type] = None

try:
    from vllm.platforms.interface import Platform as _PlatformCls

    _Platform = _PlatformCls
    _VLLM_AVAILABLE = True
except ImportError:
    logger.warning(
        "vLLM is not installed — TurboQuantPlatform will use a stub. "
        "Install vllm for full functionality."
    )

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from vllm_plugin.config import TurboQuantConfig


# ===================================================================
# TurboQuantPlatform
# ===================================================================

class TurboQuantPlatform(_Platform if _VLLM_AVAILABLE else object):  # type: ignore[misc]
    """vLLM platform plugin for TurboQuant KV cache compression.

    This class is discovered by vLLM through the ``vllm.platform_plugins``
    entry point defined in ``setup.py``.

    Attributes:
        device_type:  The device type string (``"cuda"``).
        _tq_config:   Resolved TurboQuantConfig from env vars / defaults.
    """

    device_type: str = "cuda"
    _tq_config: Optional[TurboQuantConfig] = None

    def __init__(self) -> None:
        super().__init__() if _VLLM_AVAILABLE else None
        # Lazy-init: config is created on first access
        self._tq_config = None

    # ------------------------------------------------------------------
    # Required Platform interface methods
    # ------------------------------------------------------------------

    @classmethod
    def get_attn_backend_cls(cls, *args: Any, **kwargs: Any) -> str:
        """Return the fully-qualified class name of the attention backend.

        vLLM calls this during worker initialisation to select the attention
        implementation.  We return our custom backend path.
        """
        return "vllm_plugin.attention.TurboQuantAttentionBackend"

    @classmethod
    def check_and_update_config(cls, vllm_config: Any) -> None:
        """Validate and update vLLM config for TurboQuant compatibility.

        Called by vLLM before model loading.  We enforce:
        - ``kv_cache_dtype`` must be compatible (TurboQuant bypasses paged cache).
        - ``block_size`` must be >= 16 (standard vLLM page size).
        - TurboQuant parameters are read from environment variables.
        """
        # Resolve TurboQuant config from env vars
        cls._tq_config = TurboQuantConfig()

        logger.info(
            "[TurboQuant] %s", cls._tq_config.summary()
        )

        # Warn about potential conflicts
        cache_config = getattr(vllm_config, "cache_config", None)
        if cache_config is not None:
            kv_dtype = getattr(cache_config, "kv_cache_dtype", "auto")
            if kv_dtype not in ("auto", "fp16", "float16", "bfloat16"):
                logger.warning(
                    "[TurboQuant] kv_cache_dtype=%s may conflict with "
                    "TurboQuant compression. TurboQuant manages its own cache "
                    "independently of vLLM's paged KV cache.",
                    kv_dtype,
                )

        # Ensure attention backend is set correctly
        attn_config = getattr(vllm_config, "attention_config", None)
        if attn_config is not None:
            backend = getattr(attn_config, "backend", None)
            if backend is not None and backend != "turboquant":
                logger.warning(
                    "[TurboQuant] attention_backend=%r overrides requested "
                    "backend. Forcing 'turboquant'.",
                    backend,
                )

    # ------------------------------------------------------------------
    # TurboQuant-specific helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_tq_config(cls) -> TurboQuantConfig:
        """Return the resolved TurboQuantConfig (lazily initialised)."""
        if cls._tq_config is None:
            cls._tq_config = TurboQuantConfig()
        return cls._tq_config

    @staticmethod
    def is_available() -> bool:
        """Check whether vLLM is installed and CUDA is accessible."""
        if not _VLLM_AVAILABLE:
            return False
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False
