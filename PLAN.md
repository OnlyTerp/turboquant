# TurboQuant Implementation — Bulletproof Plan
_Created 2026-03-24 ~10:30 PM EDT_

## Current State
- 5 source files written, demo runs E2E on CPU
- Cosine similarity: 0.43 (UNACCEPTABLE — needs >0.95)
- Research complete: 420 lines in RESEARCH.md, QJL reference code found
- vLLM integration uses monkey-patching (needs plugin rewrite)

## Root Cause Analysis: Why Cosine Sim = 0.43

### Bug 1: Double scaling in attention
The attention kernel computes:
```
score = (score_pq + score_qjl) * inv_sqrt_d
```
But `score_pq = q @ k_hat.t()` where `k_hat` is the FULL reconstructed vector (already scaled by norm). The PQ score already contains the full magnitude. Then dividing by sqrt(d) is correct for standard attention scaling.

BUT the QJL correction formula from the paper is:
```
<q, k> ≈ <q, k̂_pq> + (√(π/2) / d) · <S·q, signs> · ‖r‖
```
This gives an estimator for `<q, k>` — the RAW inner product. Standard attention then divides by √d:
```
attention_score = <q, k> / √d = (<q, k̂_pq> + QJL_correction) / √d
```
This looks correct in the code. So the scaling might not be the issue.

### Bug 2: S matrix inconsistency between encode and decode
In `torch_qjl_encode`:
```python
S = torch.randint(0, 2, (d, d), generator=gen).float() * 2 - 1  # Rademacher ±1
signs = (S @ x >= 0)  # sign(S·x)
```
In `torch_turboquant_attention`:
```python
S = torch.randint(0, 2, (d, d), generator=gen).float() * 2 - 1
q_proj = q @ S.t()  # S^T · q
```
Both use the same seed so S should be identical. But `S @ x` vs `q @ S.t()` — these should give the same inner product: `(S·q)^T · signs ≈ q^T · (S^T · signs)`. 

Wait — the QJL correction is: `<S·q, signs_k> * ‖r_k‖ * √(π/2) / d`

In the attention code: `q_proj = q @ S.t()` gives `S^T · q`, not `S · q`. This is WRONG if S is not symmetric (Rademacher matrices are NOT symmetric).

**THIS IS LIKELY BUG #1: `q @ S.t()` should be `q @ S` (i.e., `S · q`, not `S^T · q`)**

### Bug 3: QJL signs may be unpacked wrong in cache.py
The `_pack_bits` and `_unpack_bits` functions handle packing signs into uint32. Need to verify the bit ordering matches between encode and the attention score computation.

### Bug 4: Value decoding uses PolarQuant only
The demo's reference comparison uses decoded values (PolarQuant only, no QJL residual for V). This is correct per the paper — QJL correction only helps inner products, not vector reconstruction. But value reconstruction quality at 2-bit PolarQuant may be insufficient on its own.

## Fix Plan

### Phase 1: Fix Accuracy Bugs (CRITICAL PATH)
1. **Fix S matrix transpose bug** in `torch_turboquant_attention()` and Triton kernel
   - Change `q_proj = q @ S.t()` → `q_proj = S @ q.t()` (or equivalently `q @ S`)
   - Verify: S·q where S is [d,d] and q is [d] → result is [d]
2. **Add numerical validation** — compare TQ attention scores vs true scores token-by-token
3. **Verify FWHT inverse** — ensure encode(decode(x)) roundtrips correctly
4. **Test with known inputs** — unit vectors, identity-like patterns

### Phase 2: Fix cache.py Integration
1. Fix dtype issues (int32 vs int64 already patched)
2. Ensure packed/unpacked bit ordering is consistent
3. Add per-token validation in debug mode

### Phase 3: Rewrite vLLM Integration as Plugin
Per RESEARCH.md §4.2, use the official vLLM plugin system:
```
vllm_turboquant_plugin/
├── vllm_turboquant_plugin/
│   ├── __init__.py
│   ├── turboquant_platform.py    # extends Platform
│   ├── turboquant_attention.py   # custom AttentionBackend
│   └── turboquant_ops.py         # kernel wrappers
└── setup.py                      # entry_points for vllm.platform_plugins
```

### Phase 4: GPU Validation
1. Run demo on 5090 with Triton kernels (not just PyTorch fallbacks)
2. Benchmark: encode throughput, attention latency, memory usage
3. Compare against FP16 baseline on actual model (TerpBot Pro)

### Phase 5: Integration Test with TerpBot Pro
1. Hook into Ollama/vLLM serving TerpBot Pro on 5090
2. Run inference with TQ-compressed KV cache
3. Compare output quality (perplexity, task accuracy)
4. Measure memory savings and throughput

## Key References
- QJL CUDA code: github.com/amirzandieh/QJL
- QJL Triton bitpacking: QJL/qjl_kernel/new_pack.py (has working `_pack_along_last_dim` Triton kernel)
- FWHT Triton: github.com/arthurfeeney/fwht
- HadaCore: pytorch-labs/applied-ai (tensor core FWHT)
- vLLM plugin docs: docs.vllm.ai/en/latest/design/plugin_system/
- TurboQuant paper: arxiv.org/abs/2504.19874
- PolarQuant paper: arxiv.org/abs/2502.02617

## Critical Unknowns to Resolve
1. Does pre-RoPE vs post-RoPE quantization matter for our Nemotron model?
2. GQA handling — TerpBot Pro has grouped query attention
3. Attention sink tokens (first few) — should they stay FP16?
4. Optimal flush interval for buffer → compressed transition
