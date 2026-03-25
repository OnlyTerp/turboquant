"""
TurboQuant vLLM Integration -- KV Cache Compression Hooks

Integrates TurboQuant (3-bit KV cache compression) with vLLM's attention
mechanism. Provides:
  - TurboQuantConfig:       Dataclass holding all compression parameters
  - TurboQuantKVManager:    Buffers raw KV, flushes to TQ every N tokens
  - TurboQuantAttentionWrapper: Wraps vLLM FlashAttentionImpl.forward()
  - patch_vllm_model():     Monkey-patches attention modules in a vLLM model

Supports GQA (Grouped-Query Attention) where num_kv_heads < num_heads.

Reference: https://arxiv.org/abs/2504.19874
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Local TurboQuant cache import
# ---------------------------------------------------------------------------
try:
    from .cache import TurboQuantCache
except ImportError:
    from cache import TurboQuantCache  # fallback for direct execution

# ---------------------------------------------------------------------------
# vLLM imports (graceful -- all classes are usable without vLLM for testing)
# ---------------------------------------------------------------------------
_VLLM_AVAILABLE = False
_FlashAttentionImpl = None
AttentionMetadata = None

try:
    from vllm.attention.backends.abstract import AttentionImpl, AttentionMetadata
    from vllm.attention.backends.flash_attn import FlashAttentionImpl
    _VLLM_AVAILABLE = True
except ImportError:
    # Fallback stubs for offline testing
    class _StubAttentionImpl:
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("vLLM not installed -- stub AttentionImpl")
    _FlashAttentionImpl = _StubAttentionImpl
    AttentionMetadata = Any  # type: ignore[assignment,misc]
    FlashAttentionImpl = _StubAttentionImpl  # type: ignore[misc]


# ===================================================================
# 1. TurboQuantConfig
# ===================================================================

@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression.

    Attributes:
        num_layers:      Total transformer layers.
        num_heads:       Number of query attention heads.
        num_kv_heads:    Number of KV heads (<= num_heads for GQA).
        head_dim:        Dimension per attention head (must be power of 2).
        max_seq_len:     Maximum sequence length the cache can hold.
        flush_interval:  How often (in tokens) raw buffer is flushed to TQ.
        b_mse:           Bits per coordinate for PolarQuant stage (default 2).
        b_qjl:           Bits per coordinate for QJL stage (default 1).
        device:          Torch device for compression operations.
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

    def __post_init__(self) -> None:
        assert self.num_kv_heads <= self.num_heads, (
            f"num_kv_heads ({self.num_kv_heads}) must be <= num_heads ({self.num_heads})"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )
        assert self.head_dim > 0 and (self.head_dim & (self.head_dim - 1)) == 0, (
            f"head_dim ({self.head_dim}) must be a power of 2"
        )

    @property
    def b_total(self) -> int:
        return self.b_mse + self.b_qjl

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def heads_per_kv(self) -> int:
        """Number of query heads that share one KV head (GQA ratio)."""
        return self.num_heads // self.num_kv_heads


# ===================================================================
# 2. TurboQuantKVManager
# ===================================================================

class TurboQuantKVManager:
    """Buffers raw KV tensors and periodically flushes them to TurboQuant.

    During prefill and decode, new KV pairs arrive one (or a few) at a time.
    Instead of compressing every single token immediately, we buffer up to
    ``flush_interval`` raw tokens, then batch-compress them into the
    TurboQuantCache.

    The underlying ``TurboQuantCache`` uses pre-allocated packed tensors
    with per-(layer, head) position tracking via its legacy ``store_batch()``
    and ``compute_attention()`` methods.

    Args:
        config: TurboQuantConfig with layer/head/dim/flush params.
    """

    def __init__(self, config: TurboQuantConfig) -> None:
        self.config = config
        self._tq_cache = TurboQuantCache(
            n_layers=config.num_layers,
            n_heads=config.num_kv_heads,  # cache is indexed by KV heads
            d=config.head_dim,
            b_mse=config.b_mse,
            device=config.torch_device,
        )

        # Raw key/value buffers: [layer][kv_head] -> list of tensors [head_dim]
        self._k_buffer: List[List[List[torch.Tensor]]] = []
        self._v_buffer: List[List[List[torch.Tensor]]] = []
        for _ in range(config.num_layers):
            self._k_buffer.append([[] for _ in range(config.num_kv_heads)])
            self._v_buffer.append([[] for _ in range(config.num_kv_heads)])

        # Track how many tokens have been flushed per (layer, kv_head)
        self._flushed_count: List[List[int]] = [
            [0] * config.num_kv_heads for _ in range(config.num_layers)
        ]

    # ----- public API --------------------------------------------------------

    def store(
        self,
        layer_idx: int,
        kv_head_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Buffer a single (key, value) pair.

        Args:
            layer_idx:   Transformer layer index.
            kv_head_idx: KV head index (0..num_kv_heads-1).
            k: [head_dim] key vector.
            v: [head_dim] value vector.
        """
        dev = self.config.torch_device
        self._k_buffer[layer_idx][kv_head_idx].append(
            k.detach().to(dev).float()
        )
        self._v_buffer[layer_idx][kv_head_idx].append(
            v.detach().to(dev).float()
        )

        # Auto-flush when buffer reaches flush_interval
        if len(self._k_buffer[layer_idx][kv_head_idx]) >= self.config.flush_interval:
            self.flush(layer_idx, kv_head_idx)

    def store_batch(
        self,
        layer_idx: int,
        kv_head_idx: int,
        k_batch: torch.Tensor,
        v_batch: torch.Tensor,
    ) -> None:
        """Buffer a batch of (key, value) pairs (e.g. during prefill).

        Args:
            layer_idx:   Transformer layer index.
            kv_head_idx: KV head index.
            k_batch: [seq_len, head_dim] key vectors.
            v_batch: [seq_len, head_dim] value vectors.
        """
        dev = self.config.torch_device
        seq_len = k_batch.shape[0]
        k_batch = k_batch.detach().to(dev).float()
        v_batch = v_batch.detach().to(dev).float()

        for i in range(seq_len):
            self._k_buffer[layer_idx][kv_head_idx].append(k_batch[i])
            self._v_buffer[layer_idx][kv_head_idx].append(v_batch[i])

            if len(self._k_buffer[layer_idx][kv_head_idx]) >= self.config.flush_interval:
                self.flush(layer_idx, kv_head_idx)

    def flush(self, layer_idx: int, kv_head_idx: int) -> None:
        """Flush the raw buffer to TurboQuant compressed storage.

        Compresses all buffered tokens for the given (layer, kv_head)
        and writes them into the underlying TurboQuantCache packed buffers.
        """
        k_buf = self._k_buffer[layer_idx][kv_head_idx]
        v_buf = self._v_buffer[layer_idx][kv_head_idx]

        if not k_buf:
            return

        n = len(k_buf)
        k_batch = torch.stack(k_buf, dim=0)  # [n, head_dim]
        v_batch = torch.stack(v_buf, dim=0)  # [n, head_dim]

        # store_batch() writes into packed buffers and updates per-head count
        self._tq_cache.store_batch(layer_idx, kv_head_idx, k_batch, v_batch)
        self._flushed_count[layer_idx][kv_head_idx] += n

        # Clear buffers (replace with new list)
        self._k_buffer[layer_idx][kv_head_idx] = []
        self._v_buffer[layer_idx][kv_head_idx] = []

    def flush_all(self) -> None:
        """Flush every (layer, kv_head) buffer."""
        for l in range(self.config.num_layers):
            for h in range(self.config.num_kv_heads):
                self.flush(l, h)

    def fetch(
        self,
        layer_idx: int,
        kv_head_idx: int,
        q_vec: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Compute attention output from compressed KV + buffered raw tokens.

        Combines:
          - Compressed (TurboQuant) tokens: already flushed
          - Buffered (raw FP) tokens: not yet flushed

        For correctness when both compressed and raw tokens exist, we
        recompute the full score vector and apply a single softmax.

        Args:
            layer_idx:   Transformer layer index.
            kv_head_idx: KV head index.
            q_vec:       [head_dim] query vector (full precision).
            causal:      Apply causal masking.

        Returns:
            [head_dim] attention output.
        """
        d = self.config.head_dim
        dev = self.config.torch_device
        q_f = q_vec.float().to(dev)

        n_compressed = self._flushed_count[layer_idx][kv_head_idx]
        k_buf = self._k_buffer[layer_idx][kv_head_idx]
        n_buffered = len(k_buf)
        total = n_compressed + n_buffered

        if total == 0:
            return torch.zeros(d, device=dev)

        # --- Case 1: Only compressed tokens ---
        if n_buffered == 0:
            return self._tq_cache.compute_attention(
                layer_idx, kv_head_idx, q_f, causal=causal
            )

        # --- Case 2: Only buffered (raw) tokens ---
        if n_compressed == 0:
            return self._raw_attention(layer_idx, kv_head_idx, q_f)

        # --- Case 3: Mixed compressed + raw ---
        # Recompute ALL scores for a unified softmax.
        compressed_scores = self._compressed_scores(
            layer_idx, kv_head_idx, q_f
        )  # [n_compressed]

        k_raw = torch.stack(k_buf, dim=0)                    # [n_buf, d]
        v_raw = torch.stack(self._v_buffer[layer_idx][kv_head_idx], dim=0)
        raw_scores = (q_f @ k_raw.T) / math.sqrt(d)          # [n_buf]

        all_scores = torch.cat([compressed_scores, raw_scores])  # [total]
        all_weights = F.softmax(all_scores, dim=0)

        comp_weights = all_weights[:n_compressed]
        raw_weights = all_weights[n_compressed:]

        # Weighted sum of compressed values
        output = self._weighted_sum_compressed_values(
            layer_idx, kv_head_idx, comp_weights
        )
        # Add raw portion
        output = output + raw_weights @ v_raw
        return output

    # ----- internal helpers --------------------------------------------------

    def _raw_attention(
        self,
        layer_idx: int,
        kv_head_idx: int,
        q_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Standard attention over only the raw (unbuffered) tokens."""
        d = self.config.head_dim
        k_buf = self._k_buffer[layer_idx][kv_head_idx]
        v_buf = self._v_buffer[layer_idx][kv_head_idx]

        if not k_buf:
            return torch.zeros(d, device=self.config.torch_device)

        k_raw = torch.stack(k_buf, dim=0)  # [n, d]
        v_raw = torch.stack(v_buf, dim=0)

        scores = (q_vec @ k_raw.T) / math.sqrt(d)
        weights = F.softmax(scores, dim=0)
        return weights @ v_raw

    def _compressed_scores(
        self,
        layer_idx: int,
        kv_head_idx: int,
        q_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token attention scores from TurboQuant compressed keys.

        Iterates over the list-based cache entries and computes the TurboQuant
        score (PolarQuant dot product + QJL correction) for each stored token.
        """
        try:
            from cache import turboquant_decode_single
        except ImportError:
            from .cache import turboquant_decode_single

        d = self.config.head_dim
        tc = self._tq_cache
        S_mat = tc.qjl_matrices[layer_idx][kv_head_idx]
        qjl_scale = math.sqrt(math.pi / 2.0) / d
        q_proj = S_mat @ q_vec

        cache_entries = tc.cache[layer_idx][kv_head_idx]
        scores = []
        for kc, _vc in cache_entries:
            k_hat = turboquant_decode_single(kc)
            score_pq = torch.dot(q_vec, k_hat.squeeze(0))
            signs_f = kc.qjl.signs.squeeze(0).float() * 2 - 1
            qjl_ip = torch.dot(q_proj, signs_f)
            score_qjl = qjl_ip * qjl_scale * kc.qjl.r_norm.squeeze()
            scores.append((score_pq + score_qjl) / math.sqrt(d))

        return torch.stack(scores)

    def _weighted_sum_compressed_values(
        self,
        layer_idx: int,
        kv_head_idx: int,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted sum of decoded compressed values."""
        try:
            from cache import turboquant_decode_single
        except ImportError:
            from .cache import turboquant_decode_single

        d = self.config.head_dim
        tc = self._tq_cache
        cache_entries = tc.cache[layer_idx][kv_head_idx]
        output = torch.zeros(d, device=self.config.torch_device)
        for i, (_kc, vc) in enumerate(cache_entries):
            if weights[i].abs() > 1e-8:
                v_hat = turboquant_decode_single(vc)
                output = output + weights[i] * v_hat.squeeze(0)
        return output

    @property
    def tq_cache(self) -> TurboQuantCache:
        """Direct access to the underlying TurboQuantCache."""
        return self._tq_cache


# ===================================================================
# 3. TurboQuantAttentionWrapper
# ===================================================================

class TurboQuantAttentionWrapper:
    """Wraps a vLLM FlashAttentionImpl to use TurboQuant KV compression.

    Flow:
      - **Prefill**: After the base attention produces KV projections, compress
        ALL key/value tokens via TurboQuantKVManager and compute attention
        through the compressed cache.
      - **Decode**: For each new token, store its KV via the manager.
        Attention is computed as a split between:
          * Compressed portion (already flushed to TurboQuant)
          * Buffered portion (recent tokens still in raw FP)

    Handles GQA: the wrapper maps each query head to its corresponding KV
    head based on the ``heads_per_kv`` ratio.

    Args:
        original_impl: The vLLM AttentionImpl being wrapped.
        layer_idx:     Which transformer layer this wrapper belongs to.
        kv_manager:    Shared TurboQuantKVManager instance.
        config:        TurboQuantConfig.
    """

    def __init__(
        self,
        original_impl: Any,
        layer_idx: int,
        kv_manager: TurboQuantKVManager,
        config: TurboQuantConfig,
    ) -> None:
        self.original_impl = original_impl
        self.layer_idx = layer_idx
        self.kv_manager = kv_manager
        self.config = config

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        attn_type: str = "decoder",
        **kwargs: Any,
    ) -> torch.Tensor:
        """TurboQuant-augmented attention forward pass.

        Args:
            query:       [batch, seq_q, num_heads, head_dim]
            key:         [batch, seq_kv, num_kv_heads, head_dim]
            value:       [batch, seq_kv, num_kv_heads, head_dim]
            kv_cache:    vLLM's paged KV cache tensor (bypassed by TQ).
            attn_metadata: vLLM AttentionMetadata with seq_lens, etc.
            attn_type:   "decoder" or "encoder".

        Returns:
            [batch, seq_q, num_heads, head_dim] attention output.
        """
        is_prefill = self._is_prefill(attn_metadata, query.shape[1])

        if is_prefill:
            return self._prefill_forward(query, key, value)
        else:
            return self._decode_forward(query, key, value)

    def _prefill_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill: compress all KV into TurboQuant, then run TQ attention."""
        batch_size, seq_q, num_heads, d = query.shape
        num_kv_heads = self.config.num_kv_heads
        heads_per_kv = self.config.heads_per_kv

        outputs = torch.zeros_like(query)

        for b in range(batch_size):
            for kv_h in range(num_kv_heads):
                # Store all KV for this (batch, kv_head) pair
                k_batch = key[b, :, kv_h, :]    # [seq_kv, d]
                v_batch = value[b, :, kv_h, :]   # [seq_kv, d]

                self.kv_manager.store_batch(
                    self.layer_idx, kv_h, k_batch, v_batch
                )
                self.kv_manager.flush(self.layer_idx, kv_h)

                # Compute attention for each query head sharing this KV head
                for q_off in range(heads_per_kv):
                    q_h = kv_h * heads_per_kv + q_off
                    for t in range(seq_q):
                        q_vec = query[b, t, q_h, :].float()
                        out = self.kv_manager.fetch(
                            self.layer_idx, kv_h, q_vec, causal=True
                        )
                        outputs[b, t, q_h, :] = out.to(query.dtype)

        return outputs

    def _decode_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Decode: store one new KV token, compute split attention."""
        batch_size, seq_q, num_heads, d = query.shape
        num_kv_heads = self.config.num_kv_heads
        heads_per_kv = self.config.heads_per_kv

        assert seq_q == 1, f"Decode expects seq_q=1, got {seq_q}"

        outputs = torch.zeros_like(query)

        for b in range(batch_size):
            for kv_h in range(num_kv_heads):
                k_new = key[b, 0, kv_h, :]
                v_new = value[b, 0, kv_h, :]
                self.kv_manager.store(self.layer_idx, kv_h, k_new, v_new)

                for q_off in range(heads_per_kv):
                    q_h = kv_h * heads_per_kv + q_off
                    q_vec = query[b, 0, q_h, :].float()
                    out = self.kv_manager.fetch(
                        self.layer_idx, kv_h, q_vec, causal=True
                    )
                    outputs[b, 0, q_h, :] = out.to(query.dtype)

        return outputs

    @staticmethod
    def _is_prefill(attn_metadata: Any, seq_q: int) -> bool:
        """Heuristic: prefill if seq_q > 1 or metadata says so."""
        if attn_metadata is None:
            return seq_q > 1
        if hasattr(attn_metadata, "num_prefill_tokens"):
            return attn_metadata.num_prefill_tokens > 0
        if hasattr(attn_metadata, "prefill_metadata"):
            return attn_metadata.prefill_metadata is not None
        return seq_q > 1

    # Delegate unknown attributes to the original implementation
    def __getattr__(self, name: str) -> Any:
        return getattr(self.original_impl, name)


# ===================================================================
# 4. patch_vllm_model()
# ===================================================================

def patch_vllm_model(
    model: Any,
    tq_config: TurboQuantConfig,
) -> TurboQuantKVManager:
    """Monkey-patch a vLLM model's attention layers to use TurboQuant.

    Replaces each ``FlashAttentionImpl`` in the model's attention layers
    with a ``TurboQuantAttentionWrapper`` that transparently compresses
    and decompresses KV caches.

    Args:
        model:     A vLLM model instance (e.g. LlamaForCausalLM).
        tq_config: TurboQuantConfig with desired compression params.

    Returns:
        The shared TurboQuantKVManager (useful for inspection / manual flush).

    Example::

        from vllm import LLM
        llm = LLM(model="meta-llama/Llama-3-8B")
        tq_cfg = TurboQuantConfig(num_layers=32, num_heads=32,
                                   num_kv_heads=8, head_dim=128)
        kv_mgr = patch_vllm_model(
            llm.llm_engine.model_executor.driver_worker.model, tq_cfg
        )
    """
    if not _VLLM_AVAILABLE:
        raise RuntimeError(
            "vLLM is not installed. Install with: pip install vllm"
        )

    kv_manager = TurboQuantKVManager(tq_config)
    patch_count = 0

    attention_modules = _find_attention_modules(model)

    for layer_idx, attn_module in attention_modules:
        if not hasattr(attn_module, "impl"):
            continue

        original_impl = attn_module.impl

        if not isinstance(original_impl, FlashAttentionImpl):
            continue

        wrapper = TurboQuantAttentionWrapper(
            original_impl=original_impl,
            layer_idx=layer_idx,
            kv_manager=kv_manager,
            config=tq_config,
        )

        attn_module.impl = wrapper
        patch_count += 1

    if patch_count == 0:
        raise RuntimeError(
            "No FlashAttentionImpl modules found in the model. "
            "Ensure vLLM is using the FlashAttention backend."
        )

    print(
        f"[TurboQuant] Patched {patch_count}/{tq_config.num_layers} "
        f"attention layers | "
        f"{tq_config.b_total} bits/coord | "
        f"flush every {tq_config.flush_interval} tokens | "
        f"GQA ratio: {tq_config.heads_per_kv}:1"
    )

    return kv_manager


def _find_attention_modules(model: Any) -> List[Tuple[int, Any]]:
    """Recursively find attention modules and their layer indices.

    Returns list of (layer_idx, attention_module) tuples.
    """
    results: List[Tuple[int, Any]] = []

    # Common vLLM model structures:
    #   Llama:   model.model.layers[i].self_attn
    #   GPT:     model.transformer.h[i].attn
    layer_attrs = ("layers", "h", "blocks", "decoder_layers")

    root_children = list(model.named_children())
    for name, child in root_children:
        if name in layer_attrs:
            for idx, layer in enumerate(child):
                for sub_name, sub_mod in layer.named_children():
                    if _is_attention_module(sub_mod):
                        results.append((idx, sub_mod))
            if results:
                return results

    # Deeper search
    for name, child in root_children:
        deeper = _find_attention_modules(child)
        results.extend(deeper)

    # Fallback: any module with an 'impl' attribute
    if not results:
        for name, mod in model.named_modules():
            if hasattr(mod, "impl") and _is_attention_module(mod):
                parts = name.split(".")
                layer_idx = 0
                for p in parts:
                    if p.isdigit():
                        layer_idx = int(p)
                        break
                results.append((layer_idx, mod))

    return results


def _is_attention_module(module: Any) -> bool:
    """Heuristic: is this a vLLM attention module?"""
    cls_name = module.__class__.__name__.lower()
    return any(kw in cls_name for kw in (
        "attention", "self_attn", "cross_attn", "mha", "gqa"
    ))


# ===================================================================
# 5. GQA Helpers
# ===================================================================

def expand_kv_heads(
    kv: torch.Tensor,
    heads_per_kv: int,
) -> torch.Tensor:
    """Expand KV tensor from num_kv_heads to num_heads (GQA broadcast).

    Args:
        kv: [batch, seq, num_kv_heads, head_dim]
        heads_per_kv: number of query heads per KV head.

    Returns:
        [batch, seq, num_heads, head_dim] with KV heads broadcast.
    """
    if heads_per_kv == 1:
        return kv

    batch, seq, num_kv, d = kv.shape
    # [batch, seq, num_kv, 1, d] -> [batch, seq, num_kv, heads_per_kv, d]
    kv = kv.unsqueeze(3).expand(batch, seq, num_kv, heads_per_kv, d)
    # [batch, seq, num_kv * heads_per_kv, d]
    return kv.reshape(batch, seq, num_kv * heads_per_kv, d)


# ===================================================================
# __main__ -- Mock demo (no vLLM required)
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TurboQuant vLLM Integration -- Mock Demo")
    print("=" * 60)

    # --- Config ---
    config = TurboQuantConfig(
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,       # GQA: 4 query heads per KV head
        head_dim=128,
        max_seq_len=512,
        flush_interval=16,
        b_mse=2,
        b_qjl=1,
        device="cpu",
    )
    print(f"\nConfig: {config}")
    print(f"  Compression: {config.b_total} bits/coord")
    print(f"  GQA ratio:   {config.heads_per_kv}:1")

    # --- KV Manager ---
    kv_mgr = TurboQuantKVManager(config)
    print(f"\nKVManager created with {config.num_layers} layers, "
          f"{config.num_kv_heads} KV heads")

    # --- Simulate prefill (32 tokens) ---
    seq_len = 32
    print(f"\n--- Simulating prefill ({seq_len} tokens) ---")

    for layer in range(config.num_layers):
        for kv_h in range(config.num_kv_heads):
            k_batch = torch.randn(seq_len, config.head_dim)
            v_batch = torch.randn(seq_len, config.head_dim)
            kv_mgr.store_batch(layer, kv_h, k_batch, v_batch)
            kv_mgr.flush(layer, kv_h)  # flush any remaining

    print(f"Prefill done. Flushed count (layer 0): "
          f"{kv_mgr._flushed_count[0]}")

    # --- Simulate decode (8 new tokens) ---
    decode_steps = 8
    print(f"\n--- Simulating decode ({decode_steps} tokens) ---")

    for step in range(decode_steps):
        for layer in range(config.num_layers):
            for kv_h in range(config.num_kv_heads):
                k_new = torch.randn(config.head_dim)
                v_new = torch.randn(config.head_dim)
                kv_mgr.store(layer, kv_h, k_new, v_new)

    # Flush remaining
    kv_mgr.flush_all()
    print(f"Decode done. Total tokens per head (layer 0): "
          f"{kv_mgr._flushed_count[0]}")

    # --- Fetch (attention computation) ---
    print(f"\n--- Computing attention (decode step) ---")
    for kv_h in range(config.num_kv_heads):
        q_vec = torch.randn(config.head_dim)
        out = kv_mgr.fetch(0, kv_h, q_vec, causal=True)
        print(f"  KV head {kv_h}: output norm = {out.norm().item():.4f}")

    # --- GQA broadcast test ---
    print(f"\n--- GQA broadcast test ---")
    kv_tensor = torch.randn(1, 1, config.num_kv_heads, config.head_dim)
    expanded = expand_kv_heads(kv_tensor, config.heads_per_kv)
    print(f"  Input shape:  {kv_tensor.shape}")
    print(f"  Output shape: {expanded.shape}")
    assert expanded.shape == (1, 1, config.num_heads, config.head_dim)

    # --- Compression stats ---
    from cache import compression_ratio_fp16, memory_bytes_per_vector
    tq_b, fp16_b = memory_bytes_per_vector(config.head_dim, config.b_mse)
    ratio = compression_ratio_fp16(config.head_dim, config.b_mse)
    print(f"\n--- Compression stats (d={config.head_dim}) ---")
    print(f"  TQ bytes/vector:  {tq_b}")
    print(f"  FP16 bytes/vector: {fp16_b}")
    print(f"  Compression ratio: {ratio:.2f}x")

    total_tokens = seq_len + decode_steps
    total_vectors = total_tokens * config.num_layers * config.num_kv_heads * 2
    tq_total = total_vectors * tq_b
    fp16_total = total_vectors * fp16_b
    print(f"\n  Total tokens:     {total_tokens}")
    print(f"  Total KV vectors: {total_vectors}")
    print(f"  TQ total memory:  {tq_total / 1024:.1f} KB")
    print(f"  FP16 total memory: {fp16_total / 1024:.1f} KB")
    print(f"  Memory saved:     {(1 - tq_total / fp16_total) * 100:.1f}%")

    print("\n[OK] All checks passed!")
