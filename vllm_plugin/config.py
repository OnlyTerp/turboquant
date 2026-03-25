"""
TurboQuant vLLM Plugin — Configuration

Dataclass holding all TurboQuant compression parameters, with validation,
environment-variable overrides, and helper properties for GQA and device
management.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch


def _env_int(name: str, default: int) -> int:
    """Read an integer from an environment variable, falling back to *default*."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(
            f"Environment variable {name!r} must be an integer, got {val!r}"
        ) from None


def _env_str(name: str, default: str) -> str:
    """Read a string from an environment variable."""
    return os.environ.get(name, default)


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression.

    Every parameter can be overridden via an environment variable of the
    same name in UPPER_CASE with a ``TQ_`` prefix.  For example, setting
    ``TQ_B_MSE=3`` overrides ``b_mse``.

    Attributes:
        num_layers:      Total transformer layers in the model.
        num_heads:       Number of query attention heads.
        num_kv_heads:    Number of KV heads (<= num_heads for GQA).
        head_dim:        Dimension per attention head (must be power of 2).
        max_seq_len:     Maximum sequence length the cache can hold.
        flush_interval:  How often (in tokens) raw buffer is flushed to TQ.
        b_mse:           Bits per coordinate for the PolarQuant stage.
        b_qjl:           Bits per coordinate for the QJL stage.
        device:          Torch device string for compression operations.
    """

    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    max_seq_len: int = 4096
    flush_interval: int = 128
    b_mse: int = 2
    b_qjl: int = 1
    device: str = "cuda"

    # ------------------------------------------------------------------
    # Post-init: validation + env-var overrides
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # --- Environment-variable overrides (TQ_ prefix) ---
        self.num_layers = _env_int("TQ_NUM_LAYERS", self.num_layers)
        self.num_heads = _env_int("TQ_NUM_HEADS", self.num_heads)
        self.num_kv_heads = _env_int("TQ_NUM_KV_HEADS", self.num_kv_heads)
        self.head_dim = _env_int("TQ_HEAD_DIM", self.head_dim)
        self.max_seq_len = _env_int("TQ_MAX_SEQ_LEN", self.max_seq_len)
        self.flush_interval = _env_int("TQ_FLUSH_INTERVAL", self.flush_interval)
        self.b_mse = _env_int("TQ_B_MSE", self.b_mse)
        self.b_qjl = _env_int("TQ_B_QJL", self.b_qjl)
        self.device = _env_str("TQ_DEVICE", self.device)

        # --- Validation ---
        if self.num_kv_heads > self.num_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) must be <= "
                f"num_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.head_dim <= 0 or (self.head_dim & (self.head_dim - 1)) != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be a positive power of 2"
            )
        if self.flush_interval < 1:
            raise ValueError(
                f"flush_interval ({self.flush_interval}) must be >= 1"
            )
        if self.b_mse < 1:
            raise ValueError(f"b_mse ({self.b_mse}) must be >= 1")
        if self.b_qjl < 1:
            raise ValueError(f"b_qjl ({self.b_qjl}) must be >= 1")

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def b_total(self) -> int:
        """Total bits per coordinate (PolarQuant + QJL)."""
        return self.b_mse + self.b_qjl

    @property
    def torch_device(self) -> torch.device:
        """Torch device object derived from the device string."""
        return torch.device(self.device)

    @property
    def heads_per_kv(self) -> int:
        """Number of query heads that share one KV head (GQA ratio)."""
        return self.num_heads // self.num_kv_heads

    @property
    def compression_ratio(self) -> float:
        """Approximate compression ratio vs FP16."""
        fp16_bits = self.head_dim * 16
        tq_bits = self.head_dim * self.b_mse + 16 + self.head_dim * 1 + 16
        return fp16_bits / tq_bits

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"TurboQuant: {self.b_total}b/coord | "
            f"{self.num_layers}L × {self.num_kv_heads}KVh × d={self.head_dim} | "
            f"GQA {self.heads_per_kv}:1 | "
            f"flush={self.flush_interval} | "
            f"~{self.compression_ratio:.1f}× vs FP16"
        )
