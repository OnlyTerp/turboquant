# ⚡ TurboQuant

[![Tests](https://github.com/OnlyTerp/turboquant/actions/workflows/test.yml/badge.svg)](https://github.com/OnlyTerp/turboquant/actions)
[![arXiv](https://img.shields.io/badge/arXiv-2504.19874-b31b1b.svg)](https://arxiv.org/abs/2504.19874)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OnlyTerp/turboquant/blob/master/notebooks/demo.ipynb)

**Compress your LLM's KV cache by 5–7× with near-zero accuracy loss.** Run longer contexts, serve more users, use less GPU memory.

> First open-source implementation of [Google's TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026). 3.5 bits/value = near-identical quality to FP16. Provably within 2.7× of information-theoretic optimal.

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
# Install
git clone https://github.com/OnlyTerp/turboquant.git
cd turboquant
pip install -e .

# Run demo (synthetic vectors, no GPU needed)
python src/demo.py

# Run real model validation (downloads TinyLlama or Nemotron-Nano-4B)
python src/test_real_model.py
```

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
├── BENCHMARKS.md             # Detailed measurements
├── IMPLEMENTATION_NOTES.md   # Implementation decisions
└── setup.py                  # Package installation
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
