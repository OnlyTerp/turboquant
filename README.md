# ⚡ TurboQuant

[![Tests](https://github.com/OnlyTerp/turboquant/actions/workflows/test.yml/badge.svg)](https://github.com/OnlyTerp/turboquant/actions)
[![PyPI](https://img.shields.io/pypi/v/turboquant.svg)](https://pypi.org/project/turboquant/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/turboquant.svg)](https://pypi.org/project/turboquant/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.19874-b31b1b.svg)](https://arxiv.org/abs/2504.19874)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OnlyTerp/turboquant/blob/master/notebooks/demo.ipynb)

**Compress your LLM's KV cache by 5–7× with near-zero accuracy loss.** Run longer contexts, serve more users, use less GPU memory.

> First open-source implementation of [Google's TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026). 3.5 bits/value = near-identical quality to FP16. Provably within 2.7× of information-theoretic optimal.

### Verified on real hardware (Apr 2026)

| Where | Headline result |
|---|---|
| **RTX 5090 @ 32K prefill** (Qwen3.5-27B) | **1.24× faster** than FP16 — same GPU, same model, 4.9× less KV memory ([report](reports/2026-03-31-build-report.md)) |
| **RTX 5090 @ 64K+ context** | FP16 **OOMs**, TurboQuant still serves (1.5M context confirmed) |
| **RTX 5090 @ tg128 generation** | turbo2 = **71.1 tok/s**, FP16 = 70.22 tok/s — TurboQuant 2.5-bit is *faster than FP16* at decode because of the KV-cache bandwidth win |
| **Pure-PyTorch CPU demo** (random vectors, d=128) | 3.5-bit: **0.975 avg cosine**, 0.955 min — matches paper expectation ([report](reports/2026-04-17-demo-results.md)) |

## Table of contents

- 🎛️ **[Live demo on HuggingFace Space](https://huggingface.co/spaces/OnlyTerp/turboquant-demo)** — see your model's KV cache shrink in real time
- [What's new — April 2026](#whats-new--april-2026) • [Why TurboQuant?](#why-turboquant) • [How it works](#how-it-works) • [Quick start](#quick-start)
- [The 2026 KV compression landscape](#the-2026-kv-compression-landscape)
- [Integrations (vLLM, SGLang, llama.cpp, KVPress, LMCache, MLX)](#integrations)
- [Hardware support](#hardware-support) • [Decision guide](#which-compressor-do-i-actually-want) • [FAQ](FAQ.md)
- [Project structure](#project-structure) • [Citation](#citation) • [Credits](#credits--attribution)

## What's new — April 2026

TurboQuant went from an ICLR preprint to a genuinely viral topic in the past two weeks.
Jensen Huang spent most of GTC 2026 warning that **KV cache memory is the #1 bottleneck
for long-context inference**; TurboQuant is the most discussed answer. Highlights from
the past 72 hours:

- **Apr 15** — [LMCache Blog: "What is TurboQuant and why it matters for LLM inference (laymen's term)"](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/) notes that "some people on X even claim it is the most significant AI breakthrough this year."
- **Apr 15** — [vLLM PR #39890](https://github.com/vllm-project/vllm/pull/39890): official `turboquant_3bit` / `turboquant_4bit` KV-cache dtypes with grouped Triton store/decode paths. +2,164 LoC.
- **Apr 15** — [Towards AI: "Running a 35B Model Locally with TurboQuant — What's actually possible right now"](https://pub.towardsai.net/running-a-35b-model-locally-with-turboquant-whats-actually-possible-right-now-1ac5327430b0) — consumer GPU walkthrough, confirms TurboQuant stacks on top of AWQ/GGUF/NVFP4.
- **Apr 14** — [arXiv 2604.13226 "KV Packet"](https://arxiv.org/html/2604.13226v1): recomputation-free cross-session KV reuse via soft-token adapters — a natural partner to TurboQuant for RAG.
- **Apr 11** — [TriAttention (arXiv 2604.04921)](https://arxiv.org/abs/2604.04921): 10.7× KV memory reduction on AIME25 32K CoT via pre-RoPE trigonometric importance scoring. Complementary to TurboQuant (selection vs precision).
- **Apr 9** — [Low-Rank KV Attention](https://fin.ai/research/low-rank-key-value-attention-reducing-kv-cache-memory-and-maintaining-head-diversity/): 45–53% architectural KV reduction with lower test loss. Multiplies with TurboQuant.
- **Apr 6** — [Adaptive KV-Quant](https://arxiv.org/abs/2604.04722): learned per-token bit-width controller for on-device LLMs — can wrap TurboQuant as a backend.
- **Apr 2** — [SGLang PR #21954](https://github.com/sgl-project/sglang/pull/21954): NVFP4 KV cache strategy abstraction on Blackwell SM100/SM120. NVFP4 is the container; TurboQuant is the encoding — they compose.
- **Mar 25** — [SGLang PR #21419](https://github.com/sgl-project/sglang/pull/21419): `--kv-cache-dtype turboquant` with fused Triton kernels.

**ICLR 2026 poster**: [Zandieh et al., Sat Apr 25, 11:15 AM PDT](https://iclr.cc/virtual/2026/poster/10006985).

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

### 🔧 Our Implementation Results (Mistral-7B-Instruct-v0.3)

| Mode | Logit Cosine | Top-1 Match | KV Key Cosine | KV Value Cosine | Compression |
|------|-------------|-------------|---------------|-----------------|-------------|
| **3.5-bit** (default) | **0.963** | **80% (4/5)** | **0.992** | **0.988** | **4.9×** |
| 2.5-bit | 0.956 | 80% (4/5) | 0.973 | 0.961 | 7.1× |

Both modes use **two independent rotations** for outlier/regular channel subsets (Section 2.3) and **online codebooks** from actual data (Section 4.1).

**Rotation modes:** `rotation_mode="hadamard"` (default, O(d log d)) or `rotation_mode="dense"` (full random orthogonal via QR decomposition, O(d²)). Both satisfy P^T P = I exactly.

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
# vLLM (our plugin)
pip install -e ".[vllm]"
vllm serve meta-llama/Llama-3.1-8B-Instruct --attention-backend turboquant

# vLLM (upstream PR #39890)
gh pr checkout 39890 --repo vllm-project/vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --kv-cache-dtype turboquant_3bit

# SGLang (upstream PR #21419)
gh pr checkout 21419 --repo sgl-project/sglang
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
    --kv-cache-dtype turboquant

# llama.cpp (closest available today — native TurboQuant not yet upstream)
./llama-cli -m model.gguf -ctk q4_0 -ctv q4_0 -fa -c 131072
```

## The 2026 KV compression landscape

TurboQuant is the **precision** axis of KV compression. There are three axes — precision,
selection, and container — and the best 2026 stacks combine all three. Summary
(full analysis in [LANDSCAPE_2026.md](LANDSCAPE_2026.md)):

| Method | Released | Axis | KV reduction | Quality at ratio | TQ relationship |
|---|---|---|---|---|---|
| **TurboQuant** | Apr 2025 / ICLR'26 | **Precision (3.5 bpv)** | **4.9×** | **Identical FP16** (LongBench) | — |
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

| Engine | Status (Apr 17, 2026) | Entry point |
|---|---|---|
| **vLLM** | [PR #39890](https://github.com/vllm-project/vllm/pull/39890) (official modes) + our [`vllm_plugin/`](vllm_plugin/) | `--kv-cache-dtype turboquant_3bit` or `--attention-backend turboquant` |
| **SGLang** | [PR #21419](https://github.com/sgl-project/sglang/pull/21419) | `--kv-cache-dtype turboquant` |
| **llama.cpp** | DP4A flash-attn for quantized KV in [b8779](https://github.com/ggml-org/llama.cpp/releases/tag/b8779) (Apr 13). Native TQ path not yet upstream. | `-ctk q4_0 -ctv q4_0 -fa` (approximation) |
| **NVIDIA KVPress** | 0.4.0 — framework of "press" strategies | Stack TQ under ExpectedAttention / ThinK / AdaKV |
| **LMCache** | First-class; their [Apr 15 blog post](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/) is the best TurboQuant explainer | Store TQ-compressed K/V in distributed cache |
| **MLX (Apple)** | Pure-PyTorch path on MPS | `python src/demo.py` |
| **Transformers** | Monkey-patch via `past_key_values=TurboQuantCache(...)` | [`src/test_real_model.py`](src/test_real_model.py) |

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

A one-shot decision table for "I need to serve X on Y hardware, what do I set up?". Full
discussion in [LANDSCAPE_2026.md](LANDSCAPE_2026.md#decision-guide-which-compressor-do-i-actually-want).

| Scenario | Start with | Add on |
|---|---|---|
| Datacenter Blackwell, max throughput | NVFP4 weights + NVFP4 KV | — |
| Datacenter Hopper, long context (>128K) | FP8 weights + **TurboQuant 3.5-bit** | SnapKV / KVPress beyond 1M |
| Consumer Blackwell (RTX 5090 / RTX PRO 6000) | NVFP4 weights + **TurboQuant 3.5-bit** | — |
| Consumer Ada (RTX 4090/4080) | AWQ-INT4 + **TurboQuant 3.5-bit** | TriAttention for 32K+ CoT |
| Apple Silicon | MLX-INT4 + **TurboQuant** (CPU/MPS path) | — |
| On-device / edge | **TurboQuant 2.5-bit** or Adaptive KV-Quant | Token eviction |
| RAG, high cache reuse | **TurboQuant 3.5-bit** | KV Packet / LMCache |
| Long CoT reasoning | **TurboQuant 3.5-bit** | TriAttention |
| "Just give me something that works" | **TurboQuant 3.5-bit** | — |

## FAQ

Common questions and misconceptions are answered in **[FAQ.md](FAQ.md)**. Highlights:

- **"Is this a replacement for AWQ / GPTQ / GGUF?"** No — TurboQuant compresses the KV
  cache at inference time, stacking **on top of** weight quantization.
- **"Why 3.5 bits?"** It's a mode name from the paper. In practice, outlier channels get
  an extra MSE bit + 1-bit QJL residual; actual budget is ~3.25–4.6 bpv depending on the
  mode (see [BENCHMARKS.md §Current Demo Results](BENCHMARKS.md#current-demo-results)).
  Paper shows 3.5-bit matches FP16 on LongBench; 2.5-bit shows marginal degradation.
- **"Do I need to calibrate?"** No — TurboQuant is **data-oblivious** (random rotation +
  Lloyd-Max codebook are fixed at init).
- **"Does it work with RoPE / GQA / MLA / FlashAttention?"** Yes to all.
- **"What's the viral 'most significant breakthrough of the year' take?"** That's from
  the [LMCache blog (Apr 15, 2026)](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/)
  paraphrasing X/Twitter. Our read: the hype is largely earned, but TurboQuant is one of
  several 2026 breakthroughs (see [LANDSCAPE_2026.md](LANDSCAPE_2026.md)).

## Limitations

- **Reference implementation** — Pure PyTorch, not optimized for production throughput. Triton kernels are experimental.
- **CPU attention is slow** — The demo runs on CPU (~25× slower than FP16). GPU kernels needed for competitive speed.
- **Mixed-precision is approximate** — Our outlier channel detection differs from the paper's theoretically optimal two-independent-instances approach (see IMPLEMENTATION_NOTES.md).
- **Tested on 2 models** — Mistral-7B-Instruct and Nemotron-Nano-4B. More model validation needed.
- **vLLM plugin is a scaffold** — Not yet tested with actual vLLM serving.

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
