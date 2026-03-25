"""
TurboQuant vLLM Plugin — Custom Attention Backend

Implements ``TurboQuantAttentionBackend`` (an ``AttentionBackend`` subclass)
and ``TurboQuantAttentionImpl`` whose ``forward()`` method:

  1. Encodes new K, V tokens into the TurboQuant cache on prefill.
  2. Computes attention using PolarQuant scores + QJL correction.
  3. Uses a raw-FP buffer for recent tokens (configurable *flush_interval*).
  4. Handles GQA (num_kv_heads < num_heads) via head-to-kv-head mapping.

A ``MockAttentionBackend`` is provided when vLLM is not installed so that
standalone unit tests and demos can still run.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F

from vllm_plugin.config import TurboQuantConfig

# ---------------------------------------------------------------------------
# TurboQuantCache import (parent src/ directory)
# ---------------------------------------------------------------------------
_TQ_CACHE_AVAILABLE = False
_TurboQuantCache: Optional[type] = None
_turboquant_decode_single: Optional[Any] = None

try:
    # When installed alongside src/
    import sys as _sys
    from pathlib import Path as _Path

    _src_dir = str(_Path(__file__).resolve().parent.parent / "src")
    if _src_dir not in _sys.path:
        _sys.path.insert(0, _src_dir)

    from cache import (
        TurboQuantCache as _TQCls,
        turboquant_decode_single as _tq_decode,
    )

    _TurboQuantCache = _TQCls
    _turboquant_decode_single = _tq_decode
    _TQ_CACHE_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# vLLM imports — graceful fallback for standalone testing
# ---------------------------------------------------------------------------
_VLLM_AVAILABLE = False
_AttentionBackend: Optional[type] = None
_AttentionImpl: Optional[type] = None
_AttentionMetadata: Optional[type] = None
_AttentionType: Optional[Any] = None

try:
    from vllm.attention.backends.abstract import (
        AttentionBackend,
        AttentionImpl,
        AttentionMetadata,
        AttentionType,
    )
    from vllm.attention.selector import _BackendInfo

    _VLLM_AVAILABLE = True
    _AttentionBackend = AttentionBackend
    _AttentionImpl = AttentionImpl
    _AttentionMetadata = AttentionMetadata
    _AttentionType = AttentionType
except ImportError:
    # -----------------------------------------------------------------------
    # Mock stubs — enough to run unit tests and demos without vLLM
    # -----------------------------------------------------------------------
    class _MockAttentionBackend:  # type: ignore[too-many-ancestors]
        """Minimal stand-in for ``vllm.attention.backends.abstract.AttentionBackend``."""

        backend_name: str = "mock-turboquant"
        # AttentionImpl will be resolved by the impl_cls property

        @classmethod
        def get_impl_cls(cls) -> type:
            return TurboQuantAttentionImpl  # type: ignore[name-defined]

        @classmethod
        def make_metadata(cls, **kwargs: Any) -> Any:
            return kwargs

        @staticmethod
        def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
        ) -> Tuple[int, ...]:
            return (2, num_blocks, block_size, num_kv_heads, head_size)

        @staticmethod
        def swap_blocks(
            src: torch.Tensor, dst: torch.Tensor, src_to_dst: Any
        ) -> None:
            pass

        @staticmethod
        def copy_blocks(
            src: torch.Tensor, dst: torch.Tensor, src_to_dst: Any
        ) -> None:
            pass

    class _MockAttentionImpl:  # type: ignore[too-many-ancestors]
        """Minimal stand-in for ``vllm.attention.backends.abstract.AttentionImpl``."""

        def __init__(self, **kwargs: Any) -> None:
            pass

    _AttentionBackend = _MockAttentionBackend
    _AttentionImpl = _MockAttentionImpl
    _AttentionMetadata = Any  # type: ignore[assignment]
    _AttentionType = None


# ===================================================================
# TurboQuantAttentionImpl
# ===================================================================

class TurboQuantAttentionImpl(_AttentionImpl if _VLLM_AVAILABLE else object):  # type: ignore[misc]
    """Attention implementation backed by TurboQuant compressed KV cache.

    The forward pass intercepts K/V projections and routes them through the
    TurboQuant encoder before computing attention.  A raw-FP buffer absorbs
    recent tokens; once the buffer reaches *flush_interval* tokens it is
    batch-compressed into the TurboQuant cache.

    GQA is handled transparently: query heads are mapped to KV heads using
    the ``heads_per_kv`` ratio.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        # --- TurboQuant-specific ---
        tq_config: Optional[TurboQuantConfig] = None,
        layer_idx: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs) if _VLLM_AVAILABLE else None

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.layer_idx = layer_idx

        # TurboQuant state
        self.tq_config = tq_config or TurboQuantConfig(
            num_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=head_size,
        )

        # Build the TurboQuant cache if available
        self._tq_cache: Optional[Any] = None
        if _TQ_CACHE_AVAILABLE and _TurboQuantCache is not None:
            self._tq_cache = _TurboQuantCache(
                n_layers=1,  # each layer has its own impl
                n_heads=self.num_kv_heads,
                d=head_size,
                b_mse=self.tq_config.b_mse,
                device=self.tq_config.torch_device,
            )

        # Raw FP buffers: [kv_head] -> list of (k_vec, v_vec)
        self._k_buf: List[List[torch.Tensor]] = [[] for _ in range(self.num_kv_heads)]
        self._v_buf: List[List[torch.Tensor]] = [[] for _ in range(self.num_kv_heads)]

        # Track how many tokens have been flushed per kv_head
        self._flushed: List[int] = [0] * self.num_kv_heads

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[Any] = None,
        attn_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute TurboQuant-augmented attention.

        Args:
            query:  [batch, seq_q, num_heads, head_dim]
            key:    [batch, seq_kv, num_kv_heads, head_dim]
            value:  [batch, seq_kv, num_kv_heads, head_dim]
            kv_cache: vLLM paged KV cache (bypassed by TQ).
            attn_metadata: vLLM metadata.
            attn_type: Attention type enum.

        Returns:
            [batch, seq_q, num_heads, head_dim] attention output.
        """
        is_prefill = self._is_prefill(attn_metadata, query.shape[1])

        if is_prefill:
            return self._prefill_forward(query, key, value)
        else:
            return self._decode_forward(query, key, value)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _prefill_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Compress all KV into TurboQuant, then compute attention."""
        batch_size, seq_q, num_heads, d = query.shape
        num_kv_heads = self.num_kv_heads
        hpkv = self.tq_config.heads_per_kv

        outputs = torch.zeros_like(query)

        for b in range(batch_size):
            for kv_h in range(num_kv_heads):
                k_batch = key[b, :, kv_h, :]    # [seq_kv, d]
                v_batch = value[b, :, kv_h, :]   # [seq_kv, d]

                # Encode + flush
                self._store_batch(kv_h, k_batch, v_batch)
                self._flush(kv_h)

                # Compute attention for each Q head mapped to this KV head
                for q_off in range(hpkv):
                    q_h = kv_h * hpkv + q_off
                    for t in range(seq_q):
                        q_vec = query[b, t, q_h, :]
                        out = self._compute_attention(kv_h, q_vec)
                        outputs[b, t, q_h, :] = out.to(query.dtype)

        return outputs

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def _decode_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Store one new KV token and compute split attention."""
        batch_size, seq_q, num_heads, d = query.shape
        num_kv_heads = self.num_kv_heads
        hpkv = self.tq_config.heads_per_kv

        assert seq_q == 1, f"Decode expects seq_q=1, got {seq_q}"

        outputs = torch.zeros_like(query)

        for b in range(batch_size):
            for kv_h in range(num_kv_heads):
                k_new = key[b, 0, kv_h, :]
                v_new = value[b, 0, kv_h, :]
                self._store_single(kv_h, k_new, v_new)

                # Auto-flush
                if len(self._k_buf[kv_h]) >= self.tq_config.flush_interval:
                    self._flush(kv_h)

                for q_off in range(hpkv):
                    q_h = kv_h * hpkv + q_off
                    q_vec = query[b, 0, q_h, :]
                    out = self._compute_attention(kv_h, q_vec)
                    outputs[b, 0, q_h, :] = out.to(query.dtype)

        return outputs

    # ------------------------------------------------------------------
    # KV storage helpers
    # ------------------------------------------------------------------

    def _store_single(
        self, kv_head: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Buffer a single (k, v) pair into raw-FP storage."""
        dev = self.tq_config.torch_device
        self._k_buf[kv_head].append(k.detach().to(dev).float())
        self._v_buf[kv_head].append(v.detach().to(dev).float())

    def _store_batch(
        self, kv_head: int, k_batch: torch.Tensor, v_batch: torch.Tensor
    ) -> None:
        """Buffer a batch of (k, v) pairs."""
        dev = self.tq_config.torch_device
        for i in range(k_batch.shape[0]):
            self._k_buf[kv_head].append(k_batch[i].detach().to(dev).float())
            self._v_buf[kv_head].append(v_batch[i].detach().to(dev).float())

    def _flush(self, kv_head: int) -> None:
        """Flush raw-FP buffer → TurboQuant compressed cache."""
        if not self._k_buf[kv_head]:
            return

        k_stack = torch.stack(self._k_buf[kv_head], dim=0)  # [n, d]
        v_stack = torch.stack(self._v_buf[kv_head], dim=0)

        if self._tq_cache is not None:
            self._tq_cache.store_batch(
                0, kv_head, k_stack, v_stack
            )

        self._flushed[kv_head] += len(self._k_buf[kv_head])
        self._k_buf[kv_head] = []
        self._v_buf[kv_head] = []

    def flush_all(self) -> None:
        """Flush every KV head buffer."""
        for h in range(self.num_kv_heads):
            self._flush(h)

    # ------------------------------------------------------------------
    # Attention computation
    # ------------------------------------------------------------------

    def _compute_attention(
        self, kv_head: int, q_vec: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention output for a single query vector against
        compressed + buffered KV tokens for *kv_head*.

        Returns [head_dim] output.
        """
        d = self.head_size
        dev = self.tq_config.torch_device
        q_f = q_vec.float().to(dev)

        n_comp = self._flushed[kv_head]
        n_buf = len(self._k_buf[kv_head])
        total = n_comp + n_buf

        if total == 0:
            return torch.zeros(d, device=dev, dtype=q_vec.dtype)

        # --- Case 1: only compressed ---
        if n_buf == 0 and self._tq_cache is not None:
            return self._tq_cache.compute_attention(
                0, kv_head, q_f, causal=True
            ).to(q_vec.dtype)

        # --- Case 2: only buffered ---
        if n_comp == 0:
            return self._raw_attention(kv_head, q_f).to(q_vec.dtype)

        # --- Case 3: mixed compressed + buffered ---
        comp_scores = self._compressed_scores(kv_head, q_f)  # [n_comp]
        k_raw = torch.stack(self._k_buf[kv_head], dim=0)
        v_raw = torch.stack(self._v_buf[kv_head], dim=0)
        raw_scores = (q_f @ k_raw.T) / math.sqrt(d)

        all_scores = torch.cat([comp_scores, raw_scores])
        weights = F.softmax(all_scores, dim=0)

        # Weighted sum over compressed values
        output = self._weighted_sum_compressed(kv_head, weights[:n_comp])
        # Plus raw portion
        output = output + weights[n_comp:] @ v_raw
        return output.to(q_vec.dtype)

    def _raw_attention(
        self, kv_head: int, q_vec: torch.Tensor
    ) -> torch.Tensor:
        """Standard attention over raw (buffered) tokens only."""
        d = self.head_size
        k_buf = self._k_buf[kv_head]
        v_buf = self._v_buf[kv_head]
        if not k_buf:
            return torch.zeros(d, device=self.tq_config.torch_device)

        k_raw = torch.stack(k_buf, dim=0)
        v_raw = torch.stack(v_buf, dim=0)
        scores = (q_vec @ k_raw.T) / math.sqrt(d)
        weights = F.softmax(scores, dim=0)
        return weights @ v_raw

    def _compressed_scores(
        self, kv_head: int, q_vec: torch.Tensor
    ) -> torch.Tensor:
        """PolarQuant scores + QJL correction over compressed keys."""
        if self._tq_cache is None or _turboquant_decode_single is None:
            raise RuntimeError(
                "TurboQuantCache or decode function unavailable — "
                "ensure src/cache.py is importable."
            )

        d = self.head_size
        tc = self._tq_cache
        S = tc.qjl_matrices[0][kv_head]
        qjl_scale = math.sqrt(math.pi / 2.0) / d
        q_proj = S @ q_vec

        entries = tc.cache[0][kv_head]
        scores: List[torch.Tensor] = []
        for kc, _vc in entries:
            k_hat = _turboquant_decode_single(kc)
            score_pq = torch.dot(q_vec, k_hat.squeeze(0))
            signs_f = kc.qjl.signs.squeeze(0).float() * 2 - 1
            qjl_ip = torch.dot(q_proj, signs_f)
            score_qjl = qjl_ip * qjl_scale * kc.qjl.r_norm.squeeze()
            scores.append((score_pq + score_qjl) / math.sqrt(d))

        return torch.stack(scores)

    def _weighted_sum_compressed(
        self, kv_head: int, weights: torch.Tensor
    ) -> torch.Tensor:
        """Weighted sum of decoded compressed value vectors."""
        if self._tq_cache is None or _turboquant_decode_single is None:
            raise RuntimeError("TurboQuantCache unavailable")

        d = self.head_size
        entries = self._tq_cache.cache[0][kv_head]
        output = torch.zeros(d, device=self.tq_config.torch_device)
        for i, (_kc, vc) in enumerate(entries):
            if weights[i].abs() > 1e-8:
                v_hat = _turboquant_decode_single(vc)
                output = output + weights[i] * v_hat.squeeze(0)
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_prefill(attn_metadata: Optional[Any], seq_q: int) -> bool:
        """Determine whether this forward call is a prefill step."""
        if attn_metadata is None:
            return seq_q > 1
        if hasattr(attn_metadata, "num_prefill_tokens"):
            return attn_metadata.num_prefill_tokens > 0
        if hasattr(attn_metadata, "prefill_metadata"):
            return attn_metadata.prefill_metadata is not None
        return seq_q > 1


# ===================================================================
# TurboQuantAttentionBackend
# ===================================================================

class TurboQuantAttentionBackend(
    _AttentionBackend if _VLLM_AVAILABLE else object  # type: ignore[misc]
):
    """vLLM attention backend that uses TurboQuant-compressed KV cache.

    Registered via the ``vllm.platform_plugins`` entry point.  When selected
    (``--attention-backend turboquant``), every transformer layer will route
    attention through ``TurboQuantAttentionImpl``.
    """

    backend_name: str = "turboquant"

    @classmethod
    def get_impl_cls(cls) -> Type[TurboQuantAttentionImpl]:
        """Return the implementation class for this backend."""
        return TurboQuantAttentionImpl

    @classmethod
    def make_metadata(cls, **kwargs: Any) -> Any:
        """Create attention metadata from the given keyword arguments."""
        if _VLLM_AVAILABLE:
            return _AttentionMetadata(**kwargs)  # type: ignore[call-arg]
        return kwargs

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        """Return the shape of the KV cache tensor.

        TurboQuant bypasses the paged KV cache entirely, but vLLM still
        allocates a buffer.  We use the minimal 2-block placeholder shape.
        """
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src: torch.Tensor, dst: torch.Tensor, src_to_dst: Any
    ) -> None:
        """No-op: TurboQuant manages its own cache storage."""

    @staticmethod
    def copy_blocks(
        src: torch.Tensor, dst: torch.Tensor, src_to_dst: Any
    ) -> None:
        """No-op: TurboQuant manages its own cache storage."""
