# ⚡ TurboQuant

[![Tests](https://github.com/OnlyTerp/turboquant/actions/workflows/test.yml/badge.svg)](https://github.com/OnlyTerp/turboquant/actions)
[![PyPI](https://img.shields.io/pypi/v/turboquant.svg)](https://pypi.org/project/turboquant/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/turboquant.svg)](https://pypi.org/project/turboquant/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.19874-b31b1b.svg)](https://arxiv.org/abs/2504.19874)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OnlyTerp/turboquant/blob/master/notebooks/demo.ipynb)

**A data-oblivious KV-cache compressor.** Random rotation + scalar Lloyd-Max quantizer
that shrinks the KV cache to 3–4× smaller than FP16, no calibration data required.
Useful when GPU memory is the binding constraint on your serving rig.

> Open-source implementation of [Google's TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026). Provably within 2.7× of the information-theoretic distortion bound.

### Reality check (May 2026)

A lot has happened since the ICLR poster. **TurboQuant is not always the right answer**
and the community has converged on a more nuanced picture than the initial hype suggested.
Read this table before you start:

| Finding | Source | What it means for you |
|---|---|---|
| **FP8 KV cache is the best default** on Hopper/Blackwell — 2× capacity, no measurable accuracy loss, no throughput hit | [Red Hat AI / vLLM blog, May 11 2026](https://blog.vllm.ai/2026/05/11/turboquant.html) | Reach for `--kv-cache-dtype fp8` first. Only use TurboQuant when you need >2× compression and accept some throughput cost |
| **The QJL residual step often hurts at low bit widths** — MSE-only beats MSE+QJL on every model the community has tested | [tonbistudio V3 README](https://github.com/tonbistudio/turboquant-pytorch), [scos-lab 8-model benchmark](https://github.com/tonbistudio/turboquant-pytorch/issues/8) | Prefer the `*_nc` ("norm correction") variants; treat QJL as optional |
| **3-bit modes are not production-ready** for reasoning or >128K retrieval — ~20-point accuracy drops on AIME25 / LiveCodeBench at long context | [Red Hat blog](https://blog.vllm.ai/2026/05/11/turboquant.html) Figures 5–6 | Use `turboquant_4bit_nc` for production, save `3bit_nc` for memory-bound edge |
| **Skip the first and last 2 layers** — they hold disproportionate signal; quantizing them costs more accuracy than the bits it saves | Red Hat blog | Verify your engine config does this (vLLM does it for `*_nc` variants by default) |
| **K/V norm ratio predicts quality** — Qwen-class models need more bits on K than V (8/4 or 6/3) | [scos-lab #8](https://github.com/tonbistudio/turboquant-pytorch/issues/8) | Profile your model before picking a uniform bit budget |
| **Layer 0 has ~20% outlier channels**, middle layers ~4-6% | scos-lab | Per-layer dynamic outlier thresholds beat a fixed global allocation |

### Where TurboQuant still wins

| Scenario | Why TurboQuant beats FP8 |
|---|---|
| Single-card serving of 70B+ at 128K+ context | 3–4× KV reduction lets model + cache fit; FP8 only buys 2× |
| Consumer GPUs without FP8 attention (Ada, AMD older, Apple Silicon) | FP8 KV needs hardware FP8 attention to win; TurboQuant works everywhere |
| Long-context retrieval where memory bandwidth dominates | Reading ~52 bytes/token instead of 256 frees decode bandwidth |
| Edge / on-device with strict memory budgets | `4bit_nc` at <5% perplexity drop is hard to match |

## Table of contents

- 🎛️ **[Live demo on HuggingFace Space](https://huggingface.co/spaces/OnlyTerp/turboquant-demo)** — see your model's KV cache shrink in real time
- [What's new — May 2026](#whats-new--may-2026) • [Why TurboQuant?](#why-turboquant) • [How it works](#how-it-works) • [Quick start](#quick-start)
- [**Tips and tricks**](#tips-and-tricks) (new — from Red Hat eval + independent ports)
- [The 2026 KV compression landscape](#the-2026-kv-compression-landscape)
- [Integrations (vLLM, SGLang, llama.cpp, KVPress, LMCache, MLX)](#integrations)
- [Hardware support](#hardware-support) • [Decision guide](#which-compressor-do-i-actually-want) • [FAQ](FAQ.md)
- [Project structure](#project-structure) • [Citation](#citation) • [Credits](#credits--attribution)

## What's new — May 2026

TurboQuant landed in upstream vLLM, picked up several serious independent ports, and
got its first comprehensive third-party evaluation. The headline story has shifted from
"new SOTA, use it everywhere" to "useful in a well-defined regime, with caveats."

### The May 2026 must-reads

- **May 11** — [**Red Hat AI / vLLM blog: "A First Comprehensive Study of TurboQuant: Accuracy and Performance"**](https://blog.vllm.ai/2026/05/11/turboquant.html)
  by Eldar Kurtić, Michael Goin, Alexandre Marques. The single most important data point
  in the ecosystem right now: rigorous evaluation across Llama-3.3-70B, Qwen3-30B-A3B,
  MiniMax-M2.7 on long-context retrieval (MRCR) and reasoning (AIME25, GPQA, MATH500,
  LiveCodeBench-v6). Conclusion: **FP8 KV is the best default**, TurboQuant `4bit_nc` is
  the practical TQ variant, aggressive 3-bit modes drop accuracy meaningfully.
- **May 11–18** — Multiple independent evaluations ([TeqVolt](https://teqvolt.com/deep-dives/turboquant-google-6x-kv-cache-compression-3-bit-zero-loss),
  [AI Intensify](https://ai-intensify.com/turboquant-is-compression-and-performance-worth-the-hype/))
  converge on the same finding: **PolarQuant works, QJL hurts at 3 bits.** Disabling QJL
  gives better practical accuracy at the same compression ratio. Tested on PyTorch, MLX,
  and Triton implementations.
- **May 13** — [varjoranta/turboquant-vllm v0.13.5](https://github.com/varjoranta/turboquant-vllm/releases/tag/v0.13.5)
  adds end-to-end `Qwen3.6-35B-A3B-TQ3-native` support through vLLM 0.20.2 + FlashInfer
  CUTLASS MoE. First native-packed TQ checkpoint for a frontier MoE model — ~16 GB on
  disk (4.4× over BF16).
- **Apr 23** — [varjoranta v0.13.0](https://github.com/varjoranta/turboquant-vllm/releases/tag/v0.13.0)
  ships block-diagonal WHT CUDA kernel: **10.1× decode speedup** on Qwen3.6-35B-A3B-TQ3.
- **Late April** — [vLLM merged](https://docs.vllm.ai/en/v0.20.2/api/vllm/model_executor/layers/quantization/turboquant/)
  `--kv-cache-dtype turboquant_{k8v4, 4bit_nc, k3v4_nc, 3bit_nc}`. These dtype names that
  shipped supersede the earlier `tq3` / `tq4` / `pq4` proposals.

### Active community ports

| Repo | Stars | Engine | Notable |
|---|---|---|---|
| [0xSero/turboquant](https://github.com/0xSero/turboquant) | 1.1K | vLLM + Triton | First open-source vLLM integration; RTX 5090 + 8×RTX 3090 benchmarks; asymmetric 3-bit K / 2-bit V |
| [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) | 940+ | Pure PyTorch | V3 algorithm that **drops QJL** based on community findings; documented `residual_window=0` benchmark bug + correction |
| [varjoranta/turboquant-vllm](https://github.com/varjoranta/turboquant-vllm) (`turboquant-plus-vllm` on PyPI) | 60+ | vLLM + CUDA | Fused CUDA dequant kernels, MLX/Apple Silicon port, block-diagonal WHT for partial-rotary models, bs=1 GEMV kernel, kurtosis-aware mixed precision |
| [AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant) | 50+ | llama.cpp | Only llama.cpp fork that supports head_dim=256+ (Qwen3.5, Qwen3.6, Gemma 4); v1.4.2 adds MMA tensor-core acceleration (+63% TG) |
| [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) | 300+ | llama.cpp | head_dim=128 models only; fast on Llama-3 / Qwen3-14B-class |
| [scos-lab/turboquant](https://github.com/scos-lab/turboquant) | 14 | Research | 8-model benchmark suite; per-layer outlier analysis; published the K/V-norm-ratio finding |

### Native-packed checkpoints

- [`varjosoft/Qwen3.6-35B-A3B-TQ3-native`](https://huggingface.co/varjosoft/Qwen3.6-35B-A3B-TQ3-native)
  — pre-fused expert layout; loads via `turboquant-plus-vllm` on A100 80 GB or RTX PRO 6000.

### Earlier (still relevant) — April 2026

- [LMCache blog (Apr 15)](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/) — best layperson explainer of the algorithm.
- [Towards AI (Apr 15)](https://pub.towardsai.net/running-a-35b-model-locally-with-turboquant-whats-actually-possible-right-now-1ac5327430b0) — consumer-GPU walkthrough; TurboQuant stacks on top of AWQ / GGUF / NVFP4.
- [TriAttention (arXiv 2604.04921)](https://arxiv.org/abs/2604.04921) — 10.7× KV reduction on AIME25 via token selection; **stacks** with TurboQuant.
- [LRKV (Apr 9)](https://fin.ai/research/low-rank-key-value-attention-reducing-kv-cache-memory-and-maintaining-head-diversity/) — architectural 45-53% KV reduction at pretraining time.
- [Adaptive KV-Quant (arXiv 2604.04722)](https://arxiv.org/abs/2604.04722) — learned per-token bit-width for edge.
- [SGLang PR #21419](https://github.com/sgl-project/sglang/pull/21419) — first `--kv-cache-dtype turboquant` attempt; still WIP, see [INTEGRATIONS.md](INTEGRATIONS.md) for the current SGLang status.
- [SGLang PR #21954](https://github.com/sgl-project/sglang/pull/21954) — NVFP4 KV strategy abstraction on Blackwell SM100/SM120; container that TurboQuant composes with.

**ICLR 2026 poster (archive)**: [Zandieh et al., Apr 25 2026](https://iclr.cc/virtual/2026/poster/10006985).

> 📚 For the full 2026 landscape — including side-by-side comparisons with TriAttention,
> LRKV, MLA, KIVI, KVQuant, ParoQuant, NVFP4-KV, KVPress, KV Packet, SnapKV, H2O, and
> StreamingLLM — see **[LANDSCAPE_2026.md](LANDSCAPE_2026.md)**.

## Why TurboQuant?

### 📄 Paper Results (Llama-3.1-8B-Instruct, LongBench — from [the paper](https://arxiv.org/abs/2504.19874))

![KV Cache Compression Quality Comparison](assets/comparison_chart.png)

| Method | KV Bits | LongBench Avg | Needle-in-Haystack |
|--------|---------|---------------|-------------------|
| Full Precision | 16 | 50.06 | 0.997 |
| **TurboQuant** | **3.5** | **50.06** | **0.997** |
| **TurboQuant** | **2.5** | **49.44** | **0.997** |
| PolarQuant | 3.9 | 49.78 | 0.995 |
| KIVI | 3 | 48.50 | 0.981 |
| SnapKV | — | 44.57 | 0.858 |

> ⚠️ **Independent reproductions paint a more cautious picture than the paper.**
> The Red Hat AI / vLLM evaluation in May 2026 found that 3-bit modes give meaningful
> accuracy drops on reasoning + very long context, especially without disabling QJL. Use
> 4-bit `*_nc` variants for production. See
> [BENCHMARKS.md §Independent reproductions](BENCHMARKS.md#independent-reproductions-may-2026).

### 🔧 Our Implementation Results (Mistral-7B-Instruct-v0.3)

| Mode | Logit Cosine | Top-1 Match | KV Key Cosine | KV Value Cosine | Compression |
|------|-------------|-------------|---------------|-----------------|-------------|
| **3.5-bit** (default) | **0.963** | **80% (4/5)** | **0.992** | **0.988** | **4.9×** |
| 2.5-bit | 0.956 | 80% (4/5) | 0.973 | 0.961 | 7.1× |

Both modes use **two independent rotations** for outlier/regular channel subsets (Section 2.3) and **online codebooks** from actual data (Section 4.1).

**Rotation modes:** `rotation_mode="hadamard"` (default, O(d log d)) or `rotation_mode="dense"` (full random orthogonal via QR decomposition, O(d²)). Both satisfy P^T P = I exactly.

**QJL is optional.** Independent evaluations on PyTorch, MLX, and Triton consistently find
that the 1-bit QJL correction adds variance that softmax amplifies, hurting attention
quality at b ≤ 3. Pass `qjl_score_weight=0.0` (or set `enable_qjl=False`) to fall back to
pure MSE-only PolarQuant. See [the discussion in IMPLEMENTATION_NOTES.md §QJL Score Weight](IMPLEMENTATION_NOTES.md#qjl-score-weight).

## How It Works

TurboQuant is a two-stage vector quantizer that achieves near-optimal compression:

```
Input KV vector (FP16, d=128)
         │
         ▼
┌─────────────────────┐
│  Random Rotation Π   │  Hadamard + random signs
│  y = Π · x          │  O(d log d), preserves norms
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Scalar Lloyd-Max    │  Each coordinate independently
│  idx = quantize(y)   │  b bits per coordinate
│                      │  Beta dist ≈ N(0, 1/d)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  QJL Residual        │  1-bit sign quantization
│  sign(S · residual)  │  Unbiased inner products
└─────────┬───────────┘
          │
          ▼
   Compressed: (b+1) bits/coord + FP16 norm
   = ~3.25 bits/value at b=2
   = ~4.9× compression vs FP16
```

**Key insight**: After random rotation, each coordinate follows a Beta distribution that's near-independent of other coordinates. This means scalar quantization per coordinate is near-optimal — no coupling, no error compounding through deep models.

## Quick Start

```bash
# Install from PyPI
pip install turboquant
```

```python
# Use it
from turboquant import TurboQuantCache

cache = TurboQuantCache(n_layers=32, n_heads=8, d=128, b_mse=3, mixed_precision=True)
# ... drop into your attention loop; see INTEGRATIONS.md for vLLM / SGLang / llama.cpp
```

Or install from source for hacking on it:

```bash
git clone https://github.com/OnlyTerp/turboquant.git
cd turboquant
pip install -e ".[dev]"

# Run demo (synthetic vectors, no GPU needed)
python src/demo.py

# Run real model validation (downloads TinyLlama or Nemotron-Nano-4B)
python src/test_real_model.py
```

**Serving engines** — see [INTEGRATIONS.md](INTEGRATIONS.md) for full setup of each:

```bash
# vLLM (merged upstream — recommended)
pip install "vllm>=0.20.2"
vllm serve meta-llama/Llama-3.3-70B-Instruct --kv-cache-dtype turboquant_4bit_nc

# Production-grade vLLM fork with CUDA dequant kernels (10× decode speedup on MoE)
pip install turboquant-plus-vllm
vllm serve varjosoft/Qwen3.6-35B-A3B-TQ3-native --kv-cache-dtype turboquant_3bit_native

# Our self-contained plugin (no vLLM patching, works on older vLLM)
pip install -e ".[vllm]"
vllm serve meta-llama/Llama-3.1-8B-Instruct --attention-backend turboquant

# SGLang (still WIP, multiple competing PRs — see INTEGRATIONS.md)
gh pr checkout 21419 --repo sgl-project/sglang
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
    --kv-cache-dtype turboquant

# llama.cpp via AmesianX/TurboQuant (only fork with head_dim=256 support)
git clone https://github.com/AmesianX/TurboQuant && cd TurboQuant && make GGML_CUDA=1
./llama-cli -m model.gguf -ctk q4_0 -ctv q4_0 -fa -c 131072
```

## Tips and tricks

Distilled from the [Red Hat AI / vLLM evaluation (May 11, 2026)](https://blog.vllm.ai/2026/05/11/turboquant.html),
[tonbistudio V3 retrospective](https://github.com/tonbistudio/turboquant-pytorch),
[scos-lab 8-model benchmark](https://github.com/tonbistudio/turboquant-pytorch/issues/8),
and runtime data from 0xSero, varjoranta, and AmesianX. **Read this before you tune.**

### 1. Try FP8 first

```bash
vllm serve <model> --kv-cache-dtype fp8
```

If FP8 fits your memory budget, you are done. It's the best-quality / highest-throughput
KV format on every model the Red Hat team tested. TurboQuant only earns its keep when
you need >2× compression.

### 2. If TurboQuant: pick `4bit_nc`, not `3bit_nc`

```bash
vllm serve <model> --kv-cache-dtype turboquant_4bit_nc
```

`4bit_nc` is the "norm correction" variant — it skips the QJL residual and uses a per-token
norm rescale instead. Across Llama-3.3-70B, Qwen3-30B-A3B, and MiniMax-M2.7, it's
within ~1 pt of FP8 on AIME25 / GPQA / MATH500 / LiveCodeBench while giving 2.6–3.1×
KV reduction. `3bit_nc` and `k3v4_nc` drop **15–25 pts** on those same benchmarks at
≥128K context.

### 3. Disable QJL when working with the reference path

The `qjl_score_weight=1.0` paper default is *unbiased* but high-variance. Softmax amplifies
that variance, especially on heads with concentrated attention. Independent ports — PyTorch,
MLX, Triton — consistently see MSE-only beat MSE+QJL at b ≤ 3. In code:

```python
cache = TurboQuantCache(n_layers=..., n_heads=..., d=128, b_mse=3,
                        qjl_score_weight=0.0)  # disable QJL
# or set enable_qjl=False in newer revisions
```

### 4. Skip the first and last 2 layers

The first and last 2 layers carry disproportionate signal. The Red Hat evaluation found
that quantizing them costs more accuracy than the bits they save. vLLM does this
automatically for `*_nc` variants; if you roll your own attention backend, replicate it:

```python
SKIP_LAYERS = set(list(range(2)) + list(range(model.config.num_hidden_layers - 2,
                                              model.config.num_hidden_layers)))
for layer_idx in range(model.config.num_hidden_layers):
    if layer_idx in SKIP_LAYERS:
        # keep FP16 / FP8 KV for this layer
        ...
```

### 5. Profile K vs V norms before picking bits

Qwen-class models have wildly asymmetric K/V norm distributions (e.g. Qwen3.5-7B sees
a ~106× ratio between K and V channel norms; GPT-2-124M only ~6×). Models with extreme
ratios benefit from more bits on K than V. The `k8v4` and `k3v4_nc` variants exist for
exactly this reason:

```bash
# Use more bits on K when K dominates the norm budget (Qwen-family)
vllm serve Qwen/Qwen3-30B-A3B --kv-cache-dtype turboquant_k8v4
```

To profile your own model, run [`scos-lab/turboquant`](https://github.com/scos-lab/turboquant)
benchmarks and inspect the per-channel norm histogram.

### 6. Use per-layer dynamic outlier thresholds

A fixed "top-k=32 outlier channels" allocation wastes bits in the middle layers (only ~4-6%
of channels are actually outliers) and *under-allocates* layer 0 (~20% outliers). The
scos-lab dynamic threshold (channels with RMS > 3× median get the extra bits) closes
roughly 60% of the accuracy gap at the same average bpv.

This is the implementation policy in our `mixed_precision=True` path; just confirm your
`MixedPrecisionConfig` is using `mode="dynamic"`, not `mode="fixed_topk"`.

### 7. For frontier MoE models, use the native-packed checkpoint

If you're serving Qwen3.6-35B-A3B-class models, don't quantize at runtime — use the
pre-fused `varjosoft/Qwen3.6-35B-A3B-TQ3-native` checkpoint through `turboquant-plus-vllm`.
It bakes the per-expert rotation into the weight layout so the runtime kernel never sees
an unfused MoE expert, which is what lets it hit 10.1× decode speedup over BF16.

### 8. For llama.cpp, mind the head_dim

[spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)
hard-codes `head_dim=128`. It works for Llama-3, Qwen2.5, Qwen3-14B and most
GQA models — but Gemma 4, Qwen3.5, and Qwen3.6 use head_dim=256, where it fails silently
with degraded output. Use [AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant)
(v1.3.0+) for those models — it has a P1→P5 detection cascade that finds the real head_dim
even when `n_embd / n_head` lies (which it does on those architectures).

### 9. Stack with token selection for long reasoning

For >32K chain-of-thought (AIME, AIME25, math), token selection dominates precision.
[TriAttention](https://arxiv.org/abs/2604.04921), [SnapKV](https://arxiv.org/abs/2404.14469),
or NVIDIA's [KVPress](https://github.com/NVIDIA/kvpress) decide *which* tokens to retain;
TurboQuant compresses *each retained token*. The two operate on different axes — combine
them.

### 10. Don't trust headline compression numbers without verifying

The tonbistudio team retracted a "18/18 perfect generation at 5× compression" claim
[after discovering](https://github.com/tonbistudio/turboquant-pytorch) the eval loop
was running with `residual_window=0`, which silently disabled compression. Always log
the *actual* compressed token count after the run; needle-in-haystack at your target
context length is the only honest end-to-end test.

## The 2026 KV compression landscape

TurboQuant is the **precision** axis of KV compression. There are three axes — precision,
selection, and container — and the best 2026 stacks combine all three. Summary
(full analysis in [LANDSCAPE_2026.md](LANDSCAPE_2026.md)):

| Method | Released | Axis | KV reduction | Quality at ratio | TQ relationship |
|---|---|---|---|---|---|
| **FP8 KV (E4M3)** | Hopper / Blackwell HW | Hardware FP8 container | 2× | **No measurable loss** ([Red Hat eval](https://blog.vllm.ai/2026/05/11/turboquant.html)) | **Strictly better than TQ** when 2× is enough |
| **TurboQuant `4bit_nc`** | vLLM merged, May 2026 | **Precision (4 bpv, no QJL)** | 2.6–3.1× | Within ~1 pt of FP8 on long-context evals | The practical TQ mode |
| **TurboQuant `3bit_nc`** | vLLM merged, May 2026 | **Precision (3 bpv, no QJL)** | 3.5–4.0× | 15–25 pt drops on AIME25 / LCB | Edge / memory-bound only |
| TurboQuant (paper config, w/ QJL) | Apr 2025 / ICLR'26 | Precision (3.5 bpv + QJL) | 4.9× | Identical FP16 on LongBench (paper); worse in practice | The published configuration |
| [TriAttention](https://arxiv.org/abs/2604.04921) | **Apr 2026** | Token selection | **10.7×** on AIME25 32K CoT | Matches Full Attn reasoning | Orthogonal — stack |
| [Adaptive KV-Quant](https://arxiv.org/abs/2604.04722) | **Apr 2026** | Per-token bit-width | Variable {2, 4, 8, 16} | +8% vs static on edge | Wraps TQ as backend |
| [LRKV](https://fin.ai/research/low-rank-key-value-attention-reducing-kv-cache-memory-and-maintaining-head-diversity/) | **Apr 2026** | Architectural | 45–53% vs MHA | Lower test loss vs MHA | Pretraining-time; multiplies |
| [KV Packet](https://arxiv.org/html/2604.13226v1) | **Apr 2026** | Cache reuse | Recompute-free | TTFT ↓ for RAG | Orthogonal — caches TQ packets |
| [ParoQuant](https://z-lab.ai/projects/paroquant/) | ICLR'26 | **Weight** quant (INT4) | 4× on weights | +2.4% over AWQ on reasoning | Complementary: W4 × KV3.5 |
| [NVFP4 KV](https://github.com/sgl-project/sglang/pull/21954) | Apr 2026 | HW container (FP4) | 3.5× vs BF16 | Native Blackwell | TQ lives inside NVFP4 blocks |
| [KVPress](https://github.com/NVIDIA/kvpress) | NVIDIA | Framework | Varies | Varies | KVPress picks, TQ compresses |
| [MLA](https://arxiv.org/abs/2405.04434) | DeepSeek-V3 | Architectural | Latent down-projection | Shipped in production | TQ compresses the latent |
| [KIVI](https://arxiv.org/abs/2402.02750) | ICLR'24 | Precision (2.25 bpv) | 7.1× | LongBench 48.50 | Earlier baseline TQ beats |
| [KVQuant](https://arxiv.org/abs/2401.18079) | NeurIPS'24 | Precision + outlier | ~4.5× | LongBench ~49.5 | Needs calibration; TQ doesn't |
| [SnapKV / H2O / StreamingLLM](https://arxiv.org/abs/2404.14469) | 2024 | Token eviction | 4–20× | Drops on long context | Orthogonal — stack |

## Integrations

TurboQuant runs under every major serving engine. Concrete commands in
[INTEGRATIONS.md](INTEGRATIONS.md); summary:

| Engine | Status (May 2026) | Entry point |
|---|---|---|
| **vLLM (upstream)** | **Merged** — see [docs](https://docs.vllm.ai/en/v0.20.2/api/vllm/model_executor/layers/quantization/turboquant/) | `--kv-cache-dtype turboquant_{k8v4, 4bit_nc, k3v4_nc, 3bit_nc}` |
| **vLLM (production fork)** | [`turboquant-plus-vllm` on PyPI](https://pypi.org/project/turboquant-plus-vllm/) — CUDA dequant kernels, MoE support | `vllm serve varjosoft/Qwen3.6-35B-A3B-TQ3-native --kv-cache-dtype turboquant_3bit_native` |
| **vLLM (our plugin)** | [`vllm_plugin/`](vllm_plugin/) — works on vLLM ≥ 0.4.0 without patching | `--attention-backend turboquant` |
| **vLLM (Triton reference)** | [0xSero/turboquant](https://github.com/0xSero/turboquant) — 1.1K stars, RTX 5090 + 8×RTX 3090 benchmarks | `gh repo clone 0xSero/turboquant && pip install -e .` |
| **SGLang** | WIP — competing PRs [#21419](https://github.com/sgl-project/sglang/pull/21419), [#22048](https://github.com/sgl-project/sglang/pull/22048), [#23135](https://github.com/sgl-project/sglang/pull/23135); none merged as of May 2026 | `--kv-cache-dtype turboquant` (on PR branch) |
| **llama.cpp (head_dim=128)** | [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) | `-ctk tq3 -ctv tq3 -fa` |
| **llama.cpp (any head_dim)** | [AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant) v1.4.2 — only fork supporting head_dim=256+ | `-ctk tq3 -ctv tq3 -fa` |
| **PyTorch reference (QJL-free)** | [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) V3 | `pip install turboquant-pytorch` |
| **MLX (Apple Silicon)** | [varjoranta MLX port](https://github.com/varjoranta/turboquant-vllm) (Metal kernels) | `python -m turboquant.mlx.chat --model <id>` |
| **NVIDIA KVPress** | 0.4.0 — framework of "press" strategies | Stack TQ under ExpectedAttention / ThinK / AdaKV |
| **LMCache** | First-class; [their explainer](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/) is still the best layperson intro | Store TQ-compressed K/V in distributed cache |
| **Transformers (HF)** | Monkey-patch via `past_key_values=TurboQuantCache(...)` | [`src/test_real_model.py`](src/test_real_model.py) |

## Hardware support

| GPU | Arch | Status | Recommended stack |
|---|---|---|---|
| **B100 / B200 / GB200** | Blackwell SM100 | First-class | NVFP4 weights + TurboQuant KV + [FlashAttention-4](https://arxiviq.substack.com/p/flashattention-4-algorithm-and-kernel) |
| **RTX PRO 6000 Blackwell 96 GB** | Blackwell SM120 | Working (some [WSL2 workarounds](https://allenkuo.medium.com/finishing-what-we-started-gemma-4-nvfp4-on-vllm-desktop-blackwell-wsl2-b2088c34815a)) | NVFP4 weights + TurboQuant KV |
| **RTX 5090** | Blackwell SM120 | Working with workarounds | NVFP4 weights + TurboQuant KV |
| **H100 / H200** | Hopper SM90 | First-class | FP8 weights + TurboQuant KV |
| **A100** | Ampere SM80 | Fully supported | INT8 weights + TurboQuant KV |
| **RTX 4090 / 4080** | Ada SM89 | Fully supported | AWQ-INT4 + TurboQuant KV (+ TriAttention for 32K+ reasoning) |
| **AMD MI300X / MI325X** | CDNA3 | Via PyTorch / ROCm | INT8 weights + TurboQuant KV |
| **Apple M3 / M4 / M5** | Apple Silicon | PyTorch MPS path | MLX-INT4 weights + TurboQuant KV |
| **Jetson Orin / Thor** | Edge | [Adaptive KV-Quant](https://arxiv.org/abs/2604.04722) preferred | TurboQuant 2.5-bit fallback |

## Which compressor do I actually want?

A one-shot decision table for "I need to serve X on Y hardware, what do I set up?".
Updated May 2026 to reflect the Red Hat AI evaluation findings. Full discussion in
[LANDSCAPE_2026.md](LANDSCAPE_2026.md#decision-guide-which-compressor-do-i-actually-want).

| Scenario | Start with | Add on |
|---|---|---|
| **Just give me something that works** | **FP8 KV** (`--kv-cache-dtype fp8`) | — |
| Datacenter Hopper/Blackwell, memory-constrained at long context | **FP8 KV**; switch to **`turboquant_4bit_nc`** only if FP8 doesn't fit | SnapKV / KVPress beyond 1M |
| Datacenter Blackwell, max throughput | NVFP4 weights + **FP8 KV** | — |
| Single-card 70B+ at 128K+ | **`turboquant_4bit_nc`** (FP8 won't fit) | Skip first/last 2 layers |
| Consumer Blackwell (RTX 5090 / RTX PRO 6000) | NVFP4 weights + **FP8 KV** if fits, else **`turboquant_4bit_nc`** | — |
| Consumer Ada (RTX 4090 / 4080) — no FP8 attention | AWQ-INT4 + **`turboquant_4bit_nc`** | TriAttention for 32K+ CoT |
| Apple Silicon | MLX-INT4 + **TurboQuant via [varjoranta MLX port](https://github.com/varjoranta/turboquant-vllm)** | — |
| On-device / edge with strict memory | **`turboquant_3bit_nc`** (accept the accuracy hit) or Adaptive KV-Quant | Token eviction |
| RAG, high cache reuse | **`turboquant_4bit_nc`** | KV Packet / LMCache |
| Long CoT reasoning (>32K AIME-style) | **FP8 KV** (3-bit TQ drops accuracy too hard) | TriAttention for token selection |
| Qwen3 / Qwen3.5 / Qwen3.6 (asymmetric K/V norms) | **`turboquant_k8v4`** or `k3v4_nc` | head_dim=256? use [AmesianX llama.cpp](https://github.com/AmesianX/TurboQuant) |

## FAQ

Common questions and misconceptions are answered in **[FAQ.md](FAQ.md)**. Highlights:

- **"Should I use FP8 or TurboQuant?"** [FP8 is the better default in May 2026](FAQ.md#should-i-use-fp8-or-turboquant). Reach
  for TurboQuant only when you need >2× compression and accept some throughput cost.
- **"Does the QJL step actually help?"** [Usually not at b ≤ 3](FAQ.md#does-the-qjl-step-actually-help). MSE-only beats
  MSE+QJL on every model the community has tested. The `*_nc` variants exist for this reason.
- **"Is this a replacement for AWQ / GPTQ / GGUF?"** No — TurboQuant compresses the KV
  cache at inference time, stacking **on top of** weight quantization.
- **"Why 3.5 bits?"** It's a mode name from the paper. In practice, outlier channels get
  an extra MSE bit + 1-bit QJL residual; actual budget is ~3.25–4.6 bpv depending on the
  mode (see [BENCHMARKS.md §Current Demo Results](BENCHMARKS.md#current-demo-results)).
- **"Do I need to calibrate?"** No — TurboQuant is **data-oblivious** (random rotation +
  Lloyd-Max codebook are fixed at init).
- **"Does it work with RoPE / GQA / MLA / FlashAttention?"** Yes to all.
- **"Which layers should I skip from compression?"** [The first and last 2](FAQ.md#which-layers-should-i-skip-from-compression).
- **"How do I choose K vs V bits for my model?"** [Profile the K/V norm ratio](FAQ.md#how-do-i-choose-k-vs-v-bits-for-my-model). Symmetric for Llama-class,
  asymmetric (K > V) for Qwen-class.

## Limitations

- **Reference implementation** — Pure PyTorch, not optimized for production throughput. Triton kernels are experimental. Use [varjoranta CUDA kernels](https://github.com/varjoranta/turboquant-vllm) or [0xSero Triton](https://github.com/0xSero/turboquant) for real serving.
- **CPU attention is slow** — The demo runs on CPU (~25× slower than FP16). GPU kernels needed for competitive speed.
- **3-bit modes are not production-ready** — Independent evaluations consistently find unacceptable accuracy drops on reasoning + >128K retrieval. Use `4bit_nc` or FP8 for production; reserve 3-bit for edge / memory-bound deployments.
- **QJL is on by default for backward compatibility** — But you should turn it off (`qjl_score_weight=0.0`). See [tip #3](#3-disable-qjl-when-working-with-the-reference-path).
- **Mixed-precision is approximate** — Our outlier channel detection differs from the paper's theoretically optimal two-independent-instances approach (see IMPLEMENTATION_NOTES.md). Dynamic per-layer thresholds (scos-lab finding) help close the gap.
- **Tested on 2 models** — Mistral-7B-Instruct and Nemotron-Nano-4B. For frontier-model results, see [Red Hat AI eval](https://blog.vllm.ai/2026/05/11/turboquant.html), [0xSero RTX 5090 benchmarks](https://github.com/0xSero/turboquant), and [varjoranta Qwen3.6-35B-A3B results](https://github.com/varjoranta/turboquant-vllm).
- **vLLM plugin is a scaffold** — Not yet tested at scale with actual vLLM serving. For production, use upstream `--kv-cache-dtype turboquant_*` (merged) or the `turboquant-plus-vllm` PyPI package.

## Algorithm Details

TurboQuant implements two algorithms from the paper:

### Algorithm 1: TurboQuant_mse (MSE-optimal)

1. **Random rotation**: Multiply by randomized Hadamard matrix Π
2. **Scalar quantization**: Lloyd-Max codebook for Beta distribution, applied per coordinate
3. **Store**: b-bit index per coordinate + FP16 norm
4. **Distortion bound**: MSE ≤ √(3π/2) · 4^(-b)

### Algorithm 2: TurboQuant_prod (inner product-optimal)

1. Apply TurboQuant_mse with (b-1) bits
2. Compute residual: r = x - DeQuant(Quant(x))
3. QJL: sign(S · r) where S has i.i.d. N(0,1) entries
4. **Unbiased**: E[⟨y, x̂⟩] = ⟨y, x⟩ (no systematic bias)
5. **Total**: b bits per coordinate

### Why Not Recursive Polar Transform?

The related PolarQuant paper uses recursive polar coordinates, but TurboQuant deliberately avoids this. Recursive polar transforms couple coordinates through sin/cos operations at each level, causing errors to compound through deep models (7 levels for d=128). TurboQuant's scalar approach quantizes each coordinate independently — zero coupling, zero compounding.

## Project Structure

```
turboquant/
├── src/
│   ├── cache.py              # Core algorithm (encode/decode/cache/attention)
│   ├── demo.py               # Synthetic benchmark
│   ├── test_real_model.py    # Real transformer model validation
│   ├── test_turboquant.py    # Unit tests (33 tests)
│   ├── kernels.py            # Triton GPU kernels (experimental)
│   └── lut_attention.py      # LUT-based attention (experimental)
├── vllm_plugin/              # vLLM integration scaffold
├── deploy/                   # Docker deployment assets
│
├── README.md                 # ← you are here
├── LANDSCAPE_2026.md         # Full 2026 KV-compression ecosystem survey
├── INTEGRATIONS.md           # vLLM / SGLang / llama.cpp / MLX / KVPress / LMCache
├── FAQ.md                    # Common questions & misconceptions
├── BENCHMARKS.md             # Memory tables, throughput targets, methodology
├── IMPLEMENTATION_NOTES.md   # Rotation modes, outlier channels, QJL residual
├── pseudocode.md             # Line-by-line paper pseudocode for re-implementers
├── LAUNCH.md                 # Launch kit: threads, HN post, blog outline
├── reports/
│   ├── 2026-03-31-build-report.md    # RTX 5090 Blackwell benchmarks
│   ├── 2026-04-17-demo-results.md    # Fresh mixed-precision demo results
│   ├── 2026-04-17-demo-results.json  # Raw JSON artifact
│   └── scripts/
│       ├── run_demo_modes.py         # Reproducible 2.5-bit / 3.5-bit demo runner
│       └── check_thresholds.py       # CI gate on cosine-similarity regression
├── .github/workflows/
│   ├── test.yml              # Multi-python CI (3.10 / 3.11 / 3.12)
│   └── link-check.yml        # Weekly + PR link rot detection
└── setup.py                  # Package installation (exposes turboquant)
```

## Citation

```bibtex
@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Credits & Attribution

This is an **independent open-source implementation** of the TurboQuant algorithm. All credit for the algorithm design, theoretical analysis, and original research belongs to the paper authors.

- **Paper**: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — Published at **ICLR 2026**
- **Authors**: Amir Zandieh (Google Research), Majid Daliri (NYU), Majid Hadian (Google DeepMind), Vahab Mirrokni (Google Research)
- **Related work**: [PolarQuant](https://arxiv.org/abs/2502.02617), [QJL](https://arxiv.org/abs/2406.03482) by overlapping authors
- **Implementation**: [Terp AI Labs](https://github.com/OnlyTerp)

This implementation is not affiliated with or endorsed by Google Research, Google DeepMind, or NYU. We built it from the public paper to make TurboQuant accessible to the open-source community.

## License

MIT — see [LICENSE](LICENSE) for details.
