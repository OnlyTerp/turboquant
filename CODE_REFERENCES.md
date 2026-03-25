# TurboQuant — Code References & Patterns
_Compiled 2026-03-25 from GitHub source code research_

## Table of Contents
1. [vLLM Attention Architecture (Critical Path)](#1-vllm-attention-architecture)
2. [vLLM KV Cache Write Path (reshape_and_cache_flash)](#2-vllm-kv-cache-write-path)
3. [vLLM FP8 KV Cache Quantization Pattern](#3-vllm-fp8-kv-cache-pattern)
4. [Triton FWHT / Hadamard Transform](#4-triton-fwht-hadamard)
5. [Triton INT4 Quantization & Bitpacking Patterns](#5-triton-int4-quantization)
6. [QJL — 1-Bit Quantized JL Transform for KV Cache](#6-qjl-implementation)
7. [PolarQuant — Polar Transform KV Quantization](#7-polarquant)
8. [GemLite — Low-Bit Triton Matmul Kernels](#8-gemlite)
9. [Triton PRNG / Philox for Random Projections](#9-triton-prng-philox)
10. [0xSero TurboQuant Search Results](#10-0xsero-turboquant)
11. [Additional KV Cache Quantization Repos](#11-additional-repos)

---

## 1. vLLM Attention Architecture

### Source: `vllm/v1/attention/backends/flash_attn.py`
**URL:** `https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py`

### FlashAttentionImpl.forward() — THE critical signature to hook

```python
class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        # ... (alibi, sliding window, soft cap setup)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,       # [num_tokens, num_heads, head_size]
        key: torch.Tensor,         # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,       # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,    # [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
```

### Key observations for hooking:
1. **KV cache shape**: `[2, num_blocks, block_size, num_kv_heads, head_size]` — first dim is K/V split
2. **Cache unbind**: `key_cache, value_cache = kv_cache.unbind(0)`
3. **FP8 handling**: When `kv_cache_dtype.startswith("fp8")`, it does `key_cache = key_cache.view(dtype)`
4. **Descale pattern**: `q_descale = layer._q_scale.expand(descale_shape)` where shape is `(num_seqs, num_kv_heads)`
5. **The actual attention call**:

```python
flash_attn_varlen_func(
    q=query[:num_actual_tokens],
    k=key_cache,           # reads from CACHE, not raw key
    v=value_cache,         # reads from CACHE, not raw value
    out=output[:num_actual_tokens],
    cu_seqlens_q=cu_seqlens_q,
    max_seqlen_q=max_seqlen_q,
    seqused_k=seqused_k,
    max_seqlen_k=max_seqlen_k,
    softmax_scale=self.scale,
    causal=attn_metadata.causal,
    block_table=block_table,
    fa_version=self.vllm_flash_attn_version,
    q_descale=q_descale,
    k_descale=k_descale,
    v_descale=v_descale,
    # ... other params
)
```

### KV Cache Update (separate from forward!):

```python
def do_kv_cache_update(
    self,
    layer: torch.nn.Module,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    key_cache, value_cache = kv_cache.unbind(0)
    reshape_and_cache_flash(
        key, value,
        key_cache, value_cache,
        slot_mapping,
        self.kv_cache_dtype,
        layer._k_scale,
        layer._v_scale,
    )
```

**⚡ CRITICAL INSIGHT**: In vLLM v1, `forward_includes_kv_cache_update = False` for FlashAttention. 
The KV cache write happens BEFORE attention forward, in a separate `do_kv_cache_update()` call.
This means **we can intercept the write path independently**.

---

## 2. vLLM KV Cache Write Path

### Source: `vllm/model_executor/layers/attention/attention.py`
**URL:** `https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/attention.py`

### The Attention class forward() — top-level entry point

```python
class Attention(nn.Module, AttentionLayerBase):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        # FP8 query quantization if enabled
        if self.query_quant is not None:
            if self.impl.supports_quant_query_input:
                query, _ = self.query_quant(query, self._q_scale)

        # Reshape tensors
        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_size)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_size_v)

        # KV cache update (SEPARATE from attention forward)
        if (not self.attn_backend.forward_includes_kv_cache_update
            and self.kv_sharing_target_layer_name is None
            and key is not None and value is not None):
            kv_cache_dummy_dep = unified_kv_cache_update(
                key, value, self.layer_name
            )

        # Attention forward
        unified_attention_with_output(
            query, key, value, output,
            self.layer_name,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )
```

### Scale initialization:

```python
def set_default_quant_scales(layer, register_buffer=False):
    if register_buffer:
        layer.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_q_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_prob_scale", torch.tensor(1.0, dtype=torch.float32))
    # Also keeps host copies:
    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0
```

### Dynamic scale calculation:

```python
def calc_kv_scales(self, query, key, value):
    self._q_scale.copy_(torch.abs(query).max() / self.q_range)
    self._k_scale.copy_(torch.abs(key).max() / self.k_range)
    self._v_scale.copy_(torch.abs(value).max() / self.v_range)
    self._q_scale_float = self._q_scale.item()
    self._k_scale_float = self._k_scale.item()
    self._v_scale_float = self._v_scale.item()
    self.calculate_kv_scales = False  # Only calculate once!
```

---

## 3. vLLM FP8 KV Cache Pattern

### Source: `vllm/csrc/cache_kernels.cu`
**URL:** `https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu`

### reshape_and_cache_flash_kernel — CUDA kernel for KV cache writes with quantization

```cpp
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, block_size, num_heads, head_size]
    cache_t* __restrict__ value_cache,   // same
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t block_stride, const int64_t page_stride,
    const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int block_size, const float* k_scale, const float* v_scale,
    const int kv_scale_stride)
{
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) return;

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    // Per-tensor scale (single scale for all heads):
    if (is_contiguous_heads && kv_scale_stride == 0) {
        float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
        float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;
        CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
        CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};
        vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, n_elems, threadIdx.x, blockDim.x, k_op);
        vectorize_with_alignment<VEC_SIZE>(value_src, value_dst, n_elems, threadIdx.x, blockDim.x, v_op);
    }
    // Per-head scale:
    else {
        for (int head = warp_id; head < num_heads; head += warps_per_block) {
            float k_scale_val = k_scale[head * kv_scale_stride];
            float v_scale_val = v_scale[head * kv_scale_stride];
            // ... quantize per head
        }
    }
}
```

### CopyWithScaleOp — the actual quantization operation:

```cpp
template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
    float scale;
    __device__ __forceinline__ void operator()(OutT& dst, const InT src) const {
        if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
            dst = static_cast<OutT>(src);  // No quantization
        } else {
            dst = fp8::scaled_convert<OutT, InT, kv_dt>(src, scale);  // FP8 quantization
        }
    }
};
```

### Python-level call pattern:

```python
def reshape_and_cache_flash(
    key,           # [num_tokens, num_heads, head_size]
    value,         # [num_tokens, num_heads, head_size]
    key_cache,     # [num_blocks, block_size, num_heads, head_size]
    value_cache,   # [num_blocks, block_size, num_heads, head_size]
    slot_mapping,  # [num_tokens]
    kv_cache_dtype,
    k_scale,       # [1] or [num_heads]
    v_scale,       # [1] or [num_heads]
)
```

**⚡ KEY PATTERN**: The quantization happens during the WRITE to cache, not during read.
The read path uses `k_descale`/`v_descale` in flash_attn_varlen_func to dequantize on-the-fly.

---

## 4. Triton FWHT / Hadamard Transform

### 4a. arthurfeeney/fwht — Pure Triton Implementation
**URL:** `https://github.com/arthurfeeney/fwht`

#### Build Hadamard matrix in-register (no memory loads!):

```python
# fwht/kernel/_build_hadamard.py
import triton
import triton.language as tl

@triton.jit
def build_H(SIZE: tl.constexpr, dtype: tl.constexpr):
    """
    Construct small Hadamard matrices using bitwise dot product identity:
    H_{i,j} = (-1)^{popcount(i & j)}
    """
    tl.static_assert(0 < SIZE)
    tl.static_assert(SIZE <= 16)

    i = tl.arange(0, SIZE)
    j = tl.arange(0, SIZE)
    matching_bits = (i[:, None] & j)

    bit_sum = tl.zeros_like(matching_bits)
    for i in tl.static_range(5):
        bit_sum += matching_bits & 1
        matching_bits >>= 1

    # map odd to -1, even to 1
    H = 2 * ((bit_sum % 2) == 0) - 1
    return H.cast(dtype)
```

#### Main FWHT kernel — uses Kronecker product decomposition:

```python
# fwht/kernel/_fwht_triton.py
import math
import triton
import triton.language as tl
from ._build_hadamard import build_H

@triton.jit
def fwht_256_2step_kernel(
    a: tl.tensor,
    base: tl.tensor,
    A_SIZE: tl.constexpr,
    BASE_SIZE: tl.constexpr
):
    """Apply H_{BASE^2} = kron(H_BASE, H_BASE) via two matrix multiplies."""
    batch_size: tl.constexpr = A_SIZE // (BASE_SIZE ** 2)
    ar = a.reshape(batch_size, BASE_SIZE, BASE_SIZE)
    br = base.expand_dims(0).broadcast_to(batch_size, BASE_SIZE, BASE_SIZE)
    left = tl.dot(br, ar, out_dtype=a.dtype)
    return tl.dot(left, br, out_dtype=a.dtype).reshape(A_SIZE)

@triton.autotune(configs=[triton.Config(kwargs={}, num_warps=4)], key=['WORK_SIZE'])
@triton.jit
def fwht_256_kernel(
    a_ptr,
    scale,
    IN_SIZE: tl.constexpr,
    WORK_SIZE: tl.constexpr,
    BASE_SIZE: tl.constexpr,     # Always 16
    POWER_OF_2: tl.constexpr,
):
    tl.static_assert(WORK_SIZE >= 16)
    tl.static_assert(WORK_SIZE <= (2 ** 3) * (16 ** 3))  # max 32768

    batch_idx = tl.program_id(axis=0)
    a_ptrs = a_ptr + batch_idx * IN_SIZE + (tl.arange(0, WORK_SIZE) % IN_SIZE)
    mask = tl.arange(0, WORK_SIZE) < IN_SIZE
    a = tl.load(a_ptrs, mask=mask, other=0.0)

    base = build_H(BASE_SIZE, a.dtype)  # 16x16 Hadamard, built in registers

    BASE_SIZE_SQUARED: tl.constexpr = BASE_SIZE ** 2     # 256
    BASE_SIZE_CUBED: tl.constexpr = BASE_SIZE ** 3       # 4096

    # Step 1: kron(base, base) @ a — handles 256-element blocks
    if BASE_SIZE_SQUARED <= WORK_SIZE:
        a = fwht_256_2step_kernel(a, base, WORK_SIZE, BASE_SIZE)

    # Step 2: kron(base, kron(base, base)) @ a — handles 4096-element blocks
    if BASE_SIZE_CUBED <= WORK_SIZE:
        BATCH_SIZE: tl.constexpr = WORK_SIZE // BASE_SIZE_CUBED
        mat = a.reshape(BATCH_SIZE, BASE_SIZE, BASE_SIZE_SQUARED)
        mat = tl.dot(
            base.expand_dims(0).broadcast_to(BATCH_SIZE, BASE_SIZE, BASE_SIZE),
            mat,
            out_dtype=a.dtype
        )
        a = mat.reshape(WORK_SIZE)

    # For sizes < 256: use element-wise multiply (CUDA cores)
    if WORK_SIZE < BASE_SIZE_SQUARED:
        INNER_SIZE: tl.constexpr = WORK_SIZE // BASE_SIZE
        ar = a.reshape(INNER_SIZE, BASE_SIZE)
        ar = tl.sum(ar[:, :, None] * base[None, :, :], axis=1)
        a = ar.reshape(WORK_SIZE)

    # Handle non-power-of-16 sizes with small Hadamard
    if POWER_OF_2 > 1:
        H = build_H(POWER_OF_2, a.dtype)
        mat = a.reshape(POWER_OF_2, WORK_SIZE // POWER_OF_2)
        mat = tl.sum(H[:, :, None] * mat[None, :, :], axis=1)
        a = mat.reshape(WORK_SIZE)

    tl.store(a_ptrs, a * scale, mask=mask)
```

#### Python interface with autograd:

```python
# fwht/_interface.py
class HadamardTransformAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, inplace):
        ctx._scale = scale
        ctx._inplace = inplace
        return fwht(input, scale, inplace)

    @staticmethod
    def backward(ctx, grad_output):
        return fwht(grad_output, ctx._scale, ctx._inplace), None, None

def fast_hadamard_transform(input, scale=1.0, inplace=False):
    return HadamardTransformAutograd.apply(input, scale, inplace)
```

#### Launch pattern:

```python
def fwht(a, scale=1.0, inplace=False):
    a_flat = a.view(-1, a.size(-1))
    a_size = a_flat.size(1)
    work_size = int(2 ** math.ceil(math.log2(a_size)))
    power_of_16 = power_of_16_less_than(work_size)
    power_of_2 = work_size // power_of_16

    grid = (a_flat.size(0),)  # One program per batch element
    fwht_256_kernel[grid](
        a_flat, scale,
        IN_SIZE=a_size,
        WORK_SIZE=work_size,
        BASE_SIZE=16,
        POWER_OF_2=power_of_2,
    )
    return a
```

**⚡ KEY INSIGHT**: Max supported size is `16^3 * 2^3 = 32768` elements.
For head_size=128, this is way more than enough.
The kernel builds H in registers using bitwise ops — zero memory traffic for the Hadamard matrix!

### 4b. Dao-AILab/fast-hadamard-transform — CUDA Reference
**URL:** `https://github.com/Dao-AILab/fast-hadamard-transform`
- Supports FP32, FP16, BF16 for dims up to 32768
- Implicit zero-padding for non-power-of-2
- Benchmarked on A100
- PyTorch interface via C++ extension

### 4c. HadaCore — Tensor Core Accelerated
- Uses NVIDIA Tensor Cores + FP16
- 3.5x speedup over Dao-AILab on A100
- Plans for Triton implementation (not yet available)

### 4d. Reference FWHT Algorithm (Wikipedia butterfly):

```python
def _reference_fwht(a, scale=1.0):
    """Standard iterative butterfly FWHT."""
    h = 1
    size = a.size(-1)
    while h < size:
        for i in range(0, size, h * 2):
            for j in range(i, i + h):
                x = a[..., j].clone()
                y = a[..., j + h].clone()
                a[..., j] = x + y
                a[..., j + h] = x - y
        h *= 2
    return a * scale
```

---

## 5. Triton INT4 Quantization & Bitpacking Patterns

### 5a. GPTQ-Triton Dequantization Pattern
**URL:** `https://github.com/fpgaminer/GPTQ-triton`

Core pattern: `w = (w_packed - zeros - 1) * scales`

```python
# Conceptual GPTQ dequant in Triton (from fpgaminer/GPTQ-triton)
# Weights packed as int32, each containing 8 x 4-bit values

@triton.jit
def gptq_dequant_gemm_kernel(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    M, N, K, group_size,
    # ... block sizes
):
    # Load packed int32 weights
    b_packed = tl.load(b_ptr + offsets)

    # Extract 4-bit values via bit shifting
    shift = (k_idx % 8) * 4
    b_int4 = (b_packed >> shift) & 0xF

    # Dequantize: w = (w_int4 - zero - 1) * scale
    group_idx = k_idx // group_size
    scale = tl.load(scales_ptr + group_idx * N + n_idx)
    zero = tl.load(zeros_ptr + group_idx * N + n_idx)

    b_dequant = (b_int4.to(tl.float16) - zero - 1) * scale

    # Accumulate matmul
    acc += tl.dot(a_block, b_dequant)
```

### 5b. GemLite Bitpacking Pattern (Dropbox)
**URL:** `https://github.com/dropbox/gemlite`

```python
# GemLite packing/unpacking for 4-bit
# Pack: two 4-bit values into one uint8
packed = (val_high << 4) | (val_low & 0xF)

# Unpack in Triton kernel:
val_low = packed & 0xF
val_high = (packed >> 4) & 0xF

# For 2-bit packing into uint8:
# 4 values per byte
val0 = packed & 0x3
val1 = (packed >> 2) & 0x3
val2 = (packed >> 4) & 0x3
val3 = (packed >> 6) & 0x3
```

### 5c. Generic INT4 Quantize/Dequantize Pattern for Triton:

```python
@triton.jit
def quantize_int4_kernel(
    input_ptr, output_ptr, scale_ptr, zero_ptr,
    N, group_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input_ptr + offsets, mask=mask)

    # Compute group scale
    group_id = offsets // group_size
    x_max = tl.max(tl.abs(x))  # would need group-level reduction
    scale = x_max / 7.0  # for signed int4: [-8, 7]

    # Quantize
    x_quant = tl.libdevice.round(x / scale)
    x_quant = tl.maximum(tl.minimum(x_quant, 7), -8)

    # Pack two int4 into uint8
    even = x_quant[::2] & 0xF
    odd = (x_quant[1::2] & 0xF) << 4
    packed = even | odd

    tl.store(output_ptr + offsets // 2, packed, mask=mask[::2])
    tl.store(scale_ptr + group_id, scale)
```

---

## 6. QJL — 1-Bit Quantized JL Transform for KV Cache

### Source: `github.com/amirzandieh/QJL`
**Paper:** arxiv.org/abs/2406.03482

### Architecture:
1. Apply JL transform (random projection) to key embeddings
2. Quantize to 1-bit (sign bit only!)
3. Handle outlier coordinates separately at higher precision
4. Score computation uses quantized keys + outlier correction

### Python kernel interface:

```python
# qjl_kernel/qjl_kernel.py
def qjl_quant(key_states, outlier_indices, rand_prj, outlier_sketch_dim):
    """Quantize keys using QJL transform.
    Args:
        key_states: [batch, heads, seq_len, head_dim] — the keys to quantize
        outlier_indices: indices of outlier dimensions to keep at higher precision
        rand_prj: random projection matrix (the JL transform)
        outlier_sketch_dim: dimension for outlier sketch
    Returns:
        Quantized key representation (1-bit per projected dimension)
    """
    # Dispatches to CUDA kernel based on dtype
    return cuda_qjl_quant.qjl_quant_half_half(
        key_states, outlier_indices, rand_prj, outlier_sketch_dim)

def qjl_score(key_quant, key_outlier_quant, key_norm, key_outlier_norm,
              outlier_indices, query_sketch, query_states, rand_prj):
    """Compute attention scores using quantized keys.
    Args:
        key_quant: 1-bit quantized keys (packed as int32)
        key_outlier_quant: quantized outlier dimensions
        key_norm: norms of non-outlier components
        key_outlier_norm: norms of outlier components
        outlier_indices: which dimensions are outliers
        query_sketch: JL-transformed query
        query_states: original query (for outlier correction)
        rand_prj: same random projection used for keys
    """
    return cuda_qjl_score.qjl_score_cuda_half_half(...)
```

### Quantized matmul for value dequantization:

```python
# qjl_kernel/matmul.py
def cuda_quantized_bmm_dynamic(
    group_size: int,
    fA: torch.FloatTensor,    # [B, nh, M, K] — queries/scores
    qB: torch.IntTensor,      # [B, nh, K, N//feat_per_int] — packed quantized values
    scales: torch.FloatTensor, # quantization scales
    zeros: torch.FloatTensor,  # quantization zeros
    bits: int,                 # 2 or 4
    mqa: bool = False
) -> torch.FloatTensor:
    feat_per_int = 32 // bits  # 16 for 2-bit, 8 for 4-bit
    # ... reshape and dispatch to CUDA
    result = quantization.batchedQuantizedMultiplyAccumulate_half(
        fA, qB, scales, zeros, bits, group_size, nh, mqa)
    return result.view(B, nh, M, N)
```

**⚡ KEY INSIGHT**: QJL uses CUDA kernels (not Triton) for the core operations.
The random projection + sign quantization pattern is what we need for TurboQuant,
but we should implement it in Triton for portability.

---

## 7. PolarQuant

### Source: `github.com/ericshwu/PolarQuant`
**Paper:** arxiv.org/abs/2502.02617 / arxiv.org/abs/2502.00527

### Architecture:
1. Random preconditioning of key vectors
2. Transform to polar coordinates (radius + angles)
3. Quantize each angle independently
4. QK inner product → table lookup (huge speedup!)

### Key features:
- Uses Triton with `tl.gather` (requires custom Triton build)
- Fused dequantization + QK multiplication kernel
- Works on groups of 2D sub-vectors
- Tested on A100/A800

### Setup requirements:
- Custom Triton build with `tl.gather` support
- FlashAttention 2.3.3+ or 2.7.3+
- Python 3.8/3.10, CUDA 12.1

**⚡ NOTE**: PolarQuant requires a custom Triton build, which limits portability.
The table-lookup approach for QK computation is novel but complex.

---

## 8. GemLite — Low-Bit Triton Matmul Kernels

### Source: `github.com/dropbox/gemlite`

### Supported precisions:
- 8, 4, 2, 1-bit weight quantization
- FP16, BF16, INT8, FP8 activations
- Minimum group-size: 16
- torch.compile() compatible
- Integrated with vLLM via hqq

### Key kernel variants:
1. **GEMV** — for batch=1 decoding, uses `tl.sum` + atomic add
2. **GEMM** — tensor-core based, requires 16-row padding
3. **GEMM Split-K** — for batch 2-32 decoding
4. **GEMV RevSplit-K** — best for batch=1 with packed data

### Constructor pattern:

```python
gemlite_linear = GemLiteLinear(
    W_nbits=4,           # 8, 4, 2, or 1
    group_size=128,      # any divisible by 32
    in_features=4096,
    out_features=4096,
    input_dtype=DType.FP16,
    output_dtype=DType.FP16,
    scaled_activations=False,
)
gemlite_linear.pack(W_q, scales, zeros, bias)
out = gemlite_linear(x)
```

### vLLM integration:

```python
from hqq.utils.vllm import set_vllm_onthefly_hqq_quant
set_vllm_onthefly_hqq_quant(
    weight_bits=4, group_size=128,
    quant_mode='int4_weightonly',
    skip_modules=['lm_head', 'visual', 'vision']
)
llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", ...)
```

---

## 9. Triton PRNG / Philox for Random Projections

### Triton's built-in Philox PRNG:

```python
import triton
import triton.language as tl

@triton.jit
def random_projection_kernel(
    input_ptr, output_ptr,
    seed,
    N, D_in, D_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Generate random projection on-the-fly using Philox
    # tl.randint4x returns 4 blocks of random int32
    r0, r1, r2, r3 = tl.randint4x(seed, offsets)

    # Convert to {-1, +1} Rademacher variables
    sign = 2 * ((r0 & 1).to(tl.float32)) - 1.0

    # Or use tl.rand for uniform [0, 1):
    uniform = tl.rand(seed, offsets)

    # For Gaussian projection: Box-Muller transform
    u1 = tl.rand(seed, offsets * 2)
    u2 = tl.rand(seed, offsets * 2 + 1)
    gaussian = tl.sqrt(-2.0 * tl.log(u1)) * tl.cos(2.0 * 3.14159 * u2)
```

### Triton dropout tutorial pattern (relevant for random masks):

```python
@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Seeded random — same seed + offset = reproducible
    random = tl.rand(seed, offsets)
    x_keep = random > p

    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
```

**⚡ KEY PATTERN**: `tl.rand(seed, offset)` is Philox-based, counter-mode PRNG.
Same seed + same offset = same random number. Perfect for reproducible random projections
without storing the projection matrix!

### Sparse JL Transform pattern (from Triton tutorial challenge):

```python
# Challenge from Triton docs: implement sparse JL transform
# that generates projection matrix from seed
# Matrix R where R_ij = {+1, -1, 0} with specific probabilities
# The key insight: generate R on-the-fly, never store it

@triton.jit
def sparse_jl_kernel(input_ptr, output_ptr, seed, N, D, K, BLOCK: tl.constexpr):
    # For each output dimension k:
    # output[k] = sum_d { R[k,d] * input[d] }
    # where R[k,d] is generated from (seed, k, d)
    pid = tl.program_id(0)
    k = pid  # output dimension

    acc = tl.zeros([1], dtype=tl.float32)
    for d_start in range(0, D, BLOCK):
        d_offsets = d_start + tl.arange(0, BLOCK)
        x = tl.load(input_ptr + d_offsets, mask=d_offsets < D, other=0.0)

        # Generate random signs for this (k, d) pair
        r = tl.rand(seed, k * D + d_offsets)
        signs = tl.where(r > 0.5, 1.0, -1.0)

        acc += tl.sum(signs * x)

    tl.store(output_ptr + k, acc / tl.sqrt(float(D)))
```

---

## 10. 0xSero TurboQuant Search Results

### Status: NOT FOUND as a public repo

After extensive searching:
- **`0xSero` GitHub profile**: 197 repos, primarily known for `vllm-studio` (unified local AI workstation)
- **No repo named "turboquant"** found under 0xSero
- **`cg94301/turboquant`** exists but is a trading strategy repo (unrelated)
- **Academic "TurboQuant"** paper exists: arxiv.org/abs/2504.19874 — online vector quantization for KV cache
  - Achieves quality neutrality at 3.5 bits/channel
  - Marginal degradation at 2.5 bits/channel
  - Uses random rotation + Beta distribution + optimal scalar quantizers

### Possible explanations:
1. The repo may be private or not yet published
2. It may be under a different name on 0xSero's profile
3. The "12/12 kernel tests passing" claim may have been on social media (X/Twitter) rather than a public repo

---

## 11. Additional KV Cache Quantization Repos

### iankur/vqllm — Residual Vector Quantization for KV Cache
**URL:** `https://github.com/iankur/vqllm`
- Uses residual VQ to compress KV cache
- Built with OpenAI Triton
- NeurIPS ENLSP-IV 2024 accepted
- Uses torchtune + VQ codebooks

### MiniKV — 2-bit KV Cache Compression
**URL:** `https://github.com/Supercomputing-System-AI-Lab/MiniKV`
- 80%+ compression while maintaining accuracy
- Hardware-accelerated Triton kernel for KV eviction signals
- 2-bit quantization with adaptive selection
- FlashAttention-compatible
- ACL 2025

### vLLM Extensible Per-Token Quantized KV Cache (Proposed)
**URL:** `https://github.com/vllm-project/vllm/issues/37319`
- Proposal for extensible per-token KV cache scale infrastructure
- Triton as initial target backend (self-contained attention backend)
- Control over cache write path + attention kernels

---

## Summary: Patterns for TurboQuant Implementation

### Hook Points in vLLM:
1. **KV Cache Write**: Override `do_kv_cache_update()` or replace `reshape_and_cache_flash`
2. **Attention Forward**: The `FlashAttentionImpl.forward()` reads from `key_cache`/`value_cache`
3. **Scale Management**: `layer._k_scale`, `layer._v_scale`, `layer._q_scale` are registered buffers

### Triton Kernel Requirements:
1. **Hadamard Transform**: Use `arthurfeeney/fwht` pattern — build H in registers via bitwise ops
2. **Quantization**: Pack/unpack int4 using shift + mask, similar to GPTQ-triton
3. **Random Projection**: Use `tl.rand(seed, offset)` for reproducible Rademacher/Gaussian projection
4. **Bitpacking**: Pack 8 x 4-bit → 1 x int32, or 2 x 4-bit → 1 x uint8

### Memory Layout:
- vLLM KV cache: `[2, num_blocks, block_size, num_kv_heads, head_size]`
- Per-block slot mapping via `slot_mapping` tensor
- Separate K and V caches after `unbind(0)`

### Quantization Pipeline (per token, during cache write):
```
key_raw [num_tokens, num_kv_heads, head_size]
  → Hadamard transform (head_size dim, in-place)
  → Random sign flip (Rademacher, seeded by layer+head)
  → Absmax quantize to int4 (per-group, group_size=32 or 64)
  → Bitpack 8 int4 → 1 int32
  → Store to quantized KV cache (4x smaller)
```

### Dequantization Pipeline (during attention forward):
```
key_quant [compressed format]
  → Unpack int32 → 8 x int4
  → Dequantize: float_val = int4_val * scale
  → Inverse random sign flip (same seed)
  → Inverse Hadamard transform
  → Feed to flash_attn_varlen_func
```
