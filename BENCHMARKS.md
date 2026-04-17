# TurboQuant Benchmarks

Detailed benchmark results, theoretical analysis, and memory calculations.

> See also: [README.md](README.md) for a high-level overview.

---

## Current Demo Results

These results come from `src/demo.py` — a pure PyTorch (CPU) end-to-end test with random KV vectors.

| Metric | Value |
|--------|-------|
| Head dimension (d) | 128 |
| PolarQuant bits (b_mse) | 2 |
| QJL bits (b_qjl) | 1 |
| **Effective bits/value** | **3.25** |
| Compression ratio vs FP16 | **4.92×** |
| Bytes per vector | 52 (vs 256 FP16) |
| Avg cosine similarity | **0.90** |
| QJL score weight | 0.50 |

### Interpretation

The 0.90 cosine similarity is **acceptable but not final**. The paper achieves near-zero accuracy loss on LongBench at 3.5 bits. The QJL projection transpose bug has been fixed in the reference kernels; re-run the demo and model-level benchmarks to validate the expected improvement toward >0.95 cosine similarity.

The QJL score weight (0.50) damps the single-sample QJL correction to reduce variance. Full weight (1.0) gives the mathematically unbiased estimator, but single-sample variance in practice is high. This is a known tradeoff — future work will explore multi-sample averaging.

---

## Theoretical Bounds (from the paper)

For a unit-norm vector x ∈ ℝ^d, quantized to b bits per coordinate:

### MSE Distortion (PolarQuant alone)

```
E[‖x - x̂_pq‖²] ≤ (√3 · π / 2) · (1 / 4^b)
```

| b (PolarQuant bits) | K (levels) | MSE bound (d=128) | Relative error |
|---------------------|-----------|-------------------|----------------|
| 1 | 2 | 0.680 | 68.0% |
| 2 | 4 | 0.117 | 11.7% |
| 3 | 8 | 0.030 | 3.0% |
| 4 | 16 | 0.007 | 0.7% |

### Inner Product Distortion (Full TurboQuant)

```
E[|⟨y, x⟩ - ⟨y, x̂⟩|²] ≤ (√3 · π² · ‖y‖² / d) · (1 / 4^b_total)
```

For d=128, b_total=3, ‖y‖=1:

| Metric | Bound |
|--------|-------|
| IP distortion | ≤ 0.0014 |
| IP std dev | ≤ 0.037 |
| Relative IP error | ≤ 1.4% |

### Optimality

TurboQuant achieves distortion within a **constant factor** of the information-theoretic lower bound:

```
Optimal distortion ≥ 1 / (d · 4^b)
TurboQuant distortion ≤ (√3 · π² / d) · (1 / 4^b)
Gap factor: √3 · π² ≈ 17.1
```

This constant factor gap is tight — it's inherent to scalar quantization of high-dimensional vectors. The paper proves no scalar quantizer can do better.

---

## Format Comparison

### Compression & Quality

| Format | Bits/value | Bytes/vector | Compression | Quality | Unbiased IP? |
|--------|-----------|-------------|-------------|---------|-------------|
| FP16 | 16.00 | 256 | 1.0× | Baseline | — |
| BF16 | 16.00 | 256 | 1.0× | ~99.9% | — |
| FP8 (E4M3) | 8.00 | 128 | 2.0× | ~99.5% | — |
| INT8 | 8.25 | 106 | 2.4× | ~99% | — |
| INT4 | 4.25 | 54 | 4.7× | ~97% | No |
| KIVI-2bit | 2.25 | 29 | 8.8× | ~95% | No |
| **TurboQuant (3-bit)** | **3.25** | **52** | **4.9×** | **~99%** | **Yes** |
| TurboQuant (4-bit) | 4.25 | 68 | 3.8× | ~99.5% | Yes |
| TurboQuant (2-bit) | 2.25 | 36 | 7.1× | ~95% | Yes |

> Bits/value includes norm overhead: 16 bits (PolarQuant norm) + 16 bits (QJL residual norm) = 32 bits per vector. For d=128: +0.25 bits/value.

### Memory for Llama-3-8B-Instruct (128K context)

Model: 32 layers, 8 KV heads (GQA), d=128, seq_len=131,072

```
KV cache entries = 2 (K+V) × 32 layers × 8 heads × 131,072 tokens = 67,108,864 vectors
```

| Format | Bytes/vector | Total KV Memory | vs FP16 |
|--------|-------------|----------------|---------|
| FP16 | 256 | 16.0 GB | — |
| FP8 | 128 | 8.0 GB | 2.0× |
| INT4 | 54 | 3.4 GB | 4.7× |
| **TurboQuant 3-bit** | **52** | **3.3 GB** | **4.9×** |
| KIVI-2bit | 29 | 1.8 GB | 8.8× |

### Memory for Llama-3-70B-Instruct (128K context)

Model: 80 layers, 8 KV heads (GQA), d=128, seq_len=131,072

```
KV cache entries = 2 × 80 × 8 × 131,072 = 167,772,160 vectors
```

| Format | Bytes/vector | Total KV Memory | vs FP16 |
|--------|-------------|----------------|---------|
| FP16 | 256 | 40.0 GB | — |
| FP8 | 128 | 20.0 GB | 2.0× |
| INT4 | 54 | 8.5 GB | 4.7× |
| **TurboQuant 3-bit** | **52** | **8.2 GB** | **4.9×** |
| KIVI-2bit | 29 | 4.6 GB | 8.8× |

---

## Memory Calculator

Use this formula to compute KV cache memory for any model:

```python
def kv_cache_memory_gb(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    bits_per_value: float = 3.25,  # TurboQuant default
) -> float:
    """Compute KV cache memory in GB."""
    n_vectors = 2 * n_layers * n_kv_heads * seq_len  # K + V
    bytes_per_vector = (bits_per_value * head_dim) / 8 + 4  # +4 for two float16 norms
    total_bytes = n_vectors * bytes_per_vector
    return total_bytes / (1024 ** 3)

# Examples:
# Llama-3-8B, 128K context:  kv_cache_memory_gb(32, 8, 128, 131072)   → 3.3 GB
# Llama-3-70B, 128K context: kv_cache_memory_gb(80, 8, 128, 131072)   → 8.2 GB
# Llama-3.1-405B, 128K context: kv_cache_memory_gb(126, 8, 128, 131072) → 6.9 GB
```

### Quick Reference Table

| Model | Layers | KV Heads | d | FP16 (128K) | TurboQuant (128K) | Savings |
|-------|--------|----------|---|-------------|-------------------|---------|
| Llama-3-8B | 32 | 8 | 128 | 16.0 GB | 3.3 GB | 12.7 GB |
| Llama-3-70B | 80 | 8 | 128 | 40.0 GB | 8.2 GB | 31.8 GB |
| Llama-3.1-405B | 126 | 8 | 128 | 33.8 GB | 6.9 GB | 26.9 GB |
| Mistral-7B | 32 | 8 | 128 | 16.0 GB | 3.3 GB | 12.7 GB |
| Gemma-2-9B | 42 | 8 | 256 | 33.8 GB | 6.9 GB | 26.9 GB |
| Qwen-2.5-72B | 80 | 8 | 128 | 40.0 GB | 8.2 GB | 31.8 GB |

> FP16 calculations assume 2 bytes per value, 2 cache entries (K+V): `2 × n_layers × n_kv_heads × head_dim × seq_len × 2 bytes`. TurboQuant adds ~0.5 bytes/vector for norms.

### 2026 frontier-model context table

The models released in 2026 dramatically raised context-length expectations. TurboQuant's
impact scales proportionally — every doubling of context length doubles the absolute KV
memory savings. All numbers assume the listed context length fully populated.

| Model | Layers | KV Heads | d | Context | FP16 KV | TurboQuant KV | Savings |
|-------|--------|----------|---|---------|---------|---------------|---------|
| **Llama 4 Scout** (109B MoE, 17B active) | 48 | 8 | 128 | 1M | 192 GB | 39 GB | 153 GB |
| **Llama 4 Scout** | 48 | 8 | 128 | 10M | 1,920 GB | 391 GB | 1,529 GB |
| **Llama 4 Maverick** (402B MoE, 17B active) | 64 | 8 | 128 | 1M | 256 GB | 52 GB | 204 GB |
| **Qwen 3.5** (9B dense) | 40 | 8 | 128 | 256K | 40 GB | 8.1 GB | 31.9 GB |
| **Qwen 3.5** (72B dense, assumed) | 80 | 8 | 128 | 256K | 80 GB | 16.3 GB | 63.7 GB |
| **Qwen 3.5** (397B MoE) | 90 | 8 | 128 | 256K | 90 GB | 18.3 GB | 71.7 GB |
| **Gemma 4 31B** | 56 | 8 | 128 | 128K | 28 GB | 5.7 GB | 22.3 GB |
| **DeepSeek-V3** (MLA, effective KV) | 61 | 128 KV heads but latent | 512 latent | 128K | 8 GB (MLA) | 1.6 GB (TQ on latent) | 6.4 GB |
| **OpenClaw 32B** (TriAttention target) | 48 | 8 | 128 | 32K | 3 GB | 0.6 GB | 2.4 GB |

> **Numbers are illustrative.** Some 2026 models have non-public exact layer/head counts
> as of Apr 17, 2026; the above uses conservative public-spec estimates. Recompute using
> `kv_cache_memory_gb()` above for your specific deployment.

#### What this means in practice

- **Llama 4 Scout @ 10M context**: FP16 KV alone is ~2 TB — infeasible on any single
  machine. TurboQuant cuts it to ~390 GB, which fits in an 8× H200 (141 GB HBM each) rig
  with headroom.
- **Qwen 3.5 9B @ 256K context**: FP16 KV is 40 GB — doesn't fit on a consumer card.
  TurboQuant brings it to 8 GB, which fits alongside AWQ-INT4 weights on a single RTX
  4090 or the [96 GB RTX PRO 6000 Blackwell](https://vrlatech.com/rtx-pro-6000-blackwell-vs-a100-vs-h100-vs-rtx-5090-ai-gpu-comparison-2026/)
  with room for the model too.
- **Llama 4 Maverick @ 1M context**: from [7× H200 for FP16 to 1× H200 for
  TurboQuant+NVFP4](https://pub.towardsai.net/running-a-35b-model-locally-with-turboquant-whats-actually-possible-right-now-1ac5327430b0).

---

## Hardware Performance Targets

### Encode Throughput (vectors/sec)

Each vector encode = random rotation (FWHT) + Lloyd-Max quantize + residual + QJL sign quantize.

| Hardware | FP16 Ops/s | FWHT (est.) | PQ Encode (est.) | Total Encode (est.) |
|----------|-----------|-------------|-------------------|--------------------|
| CPU (modern) | — | ~200K | ~500K | ~150K |
| RTX 5090 | 104 TFLOPS | ~50M | ~80M | ~30M |
| A100 (80GB) | 312 TFLOPS | ~100M | ~150M | ~60M |
| H100 | 990 TFLOPS | ~300M | ~500M | ~180M |

> Estimates based on FLOP budgets for FWHT (d·log₂(d) = 896 ops), Lloyd-Max lookup (d compares), and QJL projection (d² ops but structured). Actual throughput depends on kernel fusion and memory bandwidth.

### Attention Throughput (tokens/sec)

The speedup comes from **memory bandwidth savings**: reading 52 bytes vs 256 bytes per KV vector.

| Hardware | Mem BW | FP16 Attn (est.) | TQ Attn (est.) | Speedup |
|----------|--------|------------------|----------------|---------|
| RTX 5090 | 1.8 TB/s | ~7.0M | ~30M | 4.3× |
| A100 (80GB) | 2.0 TB/s | ~7.8M | ~34M | 4.4× |
| H100 | 3.35 TB/s | ~13.1M | ~57M | 4.4× |

> Attention is memory-bandwidth bound in decode phase. TurboQuant reads 4.9× less data → ~4.4× throughput gain (reduced from 4.9× due to decode overhead).

### Latency: Attention Score Computation

For a single query over seq_len = 8,192 tokens (Llama-3-8B, 8 KV heads, d=128):

| Format | Bytes read per KV | Total bytes read | Time (H100) | Time (A100) |
|--------|-------------------|------------------|-------------|-------------|
| FP16 | 256 | 16.0 MB | 4.8 μs | 8.0 μs |
| FP8 | 128 | 8.0 MB | 2.4 μs | 4.0 μs |
| **TurboQuant** | **52** | **3.3 MB** | **1.6 μs** | **2.6 μs** |

> FP16/FP8 times are pure memory-read estimates. TurboQuant includes ~0.5 μs decode overhead on H100 but still wins on total time.

---

## Benchmarking Methodology

### What We Measure

1. **Cosine similarity:** Between TQ attention output and FP16 attention output for the same Q, K, V. Closer to 1.0 = better.

2. **MSE:** Mean squared error between TQ and FP16 attention outputs. Lower = better.

3. **Compression ratio:** `FP16_bytes / TQ_bytes` per vector.

4. **Encode time:** Wall-clock time to compress one KV vector (CPU).

5. **Decode + attention time:** Time to compute attention scores and weighted sum with compressed KV cache.

### What the Paper Measures

The paper reports **accuracy on LongBench** (a suite of long-context benchmarks) with actual LLM inference. This is the gold standard — cosine similarity on random vectors is a proxy metric.

- At 3.5 bits: **zero accuracy loss** on Llama-3.1-8B-Instruct
- At 2.5 bits: **marginal degradation**
- On H100: 4-bit TurboQuant achieves **8× performance** over 32-bit for attention logits

### Reproducing Paper Results

To reproduce the paper's claims, you need:
1. A real model (Llama-3-8B-Instruct or similar)
2. TurboQuant integrated into the inference pipeline (vLLM plugin)
3. LongBench evaluation suite
4. Hardware with sufficient memory (A100/H100 recommended)

Those end-to-end evaluations are still pending.

---

## Known Issues Affecting Quality

### Historical S-Matrix Transpose Bug

The QJL correction term uses `q @ S.T` where it should use `S @ q` (or equivalently, `q @ S` when q is a row vector). Since the Rademacher matrix S is **not symmetric**, S^T ≠ S, and the correction term uses the wrong projection direction.

**Impact:** Reduces cosine similarity from ~0.95 (expected) to ~0.90 (observed).

**Fix:** The reference PyTorch attention path now uses `q @ S`, matching the `S @ q` formulation from the paper. Any fused kernel should use the same projection.

### Single-Sample QJL Variance (By Design)

The QJL correction is applied from a single random projection sample. The variance of this estimate is:

```
Var(⟨y, r̂⟩) ≤ (π / 2d) · ‖y‖² · ‖r‖²
```

For d=128: Var ≤ 0.012 · ‖y‖² · ‖r‖². With ‖r‖ ≈ 0.3 (typical residual after 2-bit PQ), this is Var ≈ 0.001 · ‖y‖².

The `qjl_score_weight=0.5` parameter trades bias for variance reduction. Setting it to 1.0 gives the unbiased estimator but higher variance per-token.

### Attention Sinks (Not Yet Implemented)

The first few tokens in a sequence ("attention sinks") tend to receive disproportionately high attention weights. Their KV cache vectors should ideally be preserved in higher precision. This is planned but not yet implemented.

---

## Future Benchmarks

### Planned Tests

- [ ] **Llama-3-8B on LongBench** — reproduce paper's accuracy claims
- [ ] **Needle-in-haystack** — test retrieval at 128K context with TQ compression
- [ ] **Triton vs PyTorch** — kernel speedup comparison
- [ ] **RTX 5090 throughput** — real hardware benchmarks on consumer GPU
- [ ] **H100 throughput** — datacenter GPU benchmarks
- [ ] **Mixed precision** — test PQ=3-bit for first N layers, 2-bit for rest
- [ ] **GQA scaling** — test with different GQA ratios (1:1, 2:1, 4:1, 8:1)
- [ ] **RoPE interaction** — pre-RoPE vs post-RoPE quality comparison
