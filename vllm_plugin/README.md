# TurboQuant vLLM Plugin

3.5-bit KV cache compression for vLLM, combining **PolarQuant** (MSE-optimal scalar quantization) with **QJL** (1-bit Quantized Johnson-Lindenstrauss residual correction).

> **Reference:** [TurboQuant paper (arxiv 2504.19874)](https://arxiv.org/abs/2504.19874)

---

## What It Does

TurboQuant compresses the transformer KV cache from **16 bits/channel (FP16) → ~3.5 bits/channel** with near-zero accuracy loss on LongBench benchmarks. The speedup comes from replacing FP32 Q×K matrix multiply with integer table lookups (LUT).

| Stage | Bits | Purpose |
|-------|------|---------|
| PolarQuant | 2 | MSE-optimal scalar quantization after randomized Hadamard rotation |
| QJL | 1 | Unbiased residual correction via sign quantization |
| **Total** | **3** | **~4.9× compression vs FP16** |

---

## Installation

```bash
cd scripts/turboquant
pip install -e .
```

Or with vLLM as an extra dependency:

```bash
pip install -e ".[vllm]"
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- vLLM ≥ 0.4.0 (optional for standalone testing)
- CUDA-capable GPU (recommended)

---

## Usage

### Quick Start

```bash
# Install the plugin
cd scripts/turboquant && pip install -e .

# Serve a model with TurboQuant KV cache compression
vllm serve meta-llama/Llama-3-8B-Instruct --attention-backend turboquant
```

The plugin is auto-discovered by vLLM through the `vllm.platform_plugins` entry point. No manual imports needed.

### Configuration via Environment Variables

All parameters can be set via `TQ_*` environment variables **before** launching vLLM:

```bash
# Core parameters
export TQ_B_MSE=2              # PolarQuant bits per coordinate (default: 2)
export TQ_B_QJL=1              # QJL bits per coordinate (default: 1)
export TQ_FLUSH_INTERVAL=128   # Raw buffer size before compression (default: 128)

# Model-specific (auto-detected when vLLM provides model config)
export TQ_NUM_LAYERS=32
export TQ_NUM_HEADS=32
export TQ_NUM_KV_HEADS=8      # For GQA models (e.g., Llama-3)
export TQ_HEAD_DIM=128

# Device
export TQ_DEVICE=cuda          # "cuda" or "cpu" (default: cuda)

# Then launch
vllm serve meta-llama/Llama-3-8B-Instruct --attention-backend turboquant
```

### Python API (Standalone / Testing)

```python
from vllm_plugin import TurboQuantConfig, TurboQuantAttentionImpl

# Configure
config = TurboQuantConfig(
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,      # Llama-3 GQA
    head_dim=128,
    flush_interval=64,
    b_mse=2,
    b_qjl=1,
    device="cuda",
)

# Create attention implementation
impl = TurboQuantAttentionImpl(
    num_heads=32,
    head_size=128,
    scale=1.0 / (128 ** 0.5),
    num_kv_heads=8,
    tq_config=config,
    layer_idx=0,
)

# Forward pass (same interface as vLLM AttentionImpl)
output = impl.forward(query, key, value)
```

---

## Architecture

```
vllm_plugin/
├── __init__.py      # Package init, version, public API
├── config.py        # TurboQuantConfig dataclass with env-var overrides
├── attention.py     # TurboQuantAttentionBackend + TurboQuantAttentionImpl
├── platform.py      # TurboQuantPlatform plugin (entry point target)
└── README.md        # This file

setup.py             # Entry point registration
src/
└── cache.py         # TurboQuantCache (PolarQuant + QJL encode/decode)
```

### Attention Flow

1. **Prefill:** Query, Key, Value tensors arrive → all KV tokens are batch-encoded into TurboQuantCache → attention computed via PQ scores + QJL correction
2. **Decode:** Single new KV token buffered in raw FP → when buffer hits `flush_interval`, batch-compress into TQ cache → attention computed over compressed + buffered tokens with unified softmax
3. **GQA:** Query heads are mapped to KV heads using `heads_per_kv` ratio (`num_heads // num_kv_heads`)

---

## Configuration Reference

| Env Variable | Parameter | Default | Description |
|---|---|---|---|
| `TQ_NUM_LAYERS` | `num_layers` | 32 | Transformer layer count |
| `TQ_NUM_HEADS` | `num_heads` | 32 | Query attention heads |
| `TQ_NUM_KV_HEADS` | `num_kv_heads` | 32 | KV heads (≤ num_heads) |
| `TQ_HEAD_DIM` | `head_dim` | 128 | Head dimension (power of 2) |
| `TQ_MAX_SEQ_LEN` | `max_seq_len` | 4096 | Max sequence length |
| `TQ_FLUSH_INTERVAL` | `flush_interval` | 128 | Buffer size before TQ flush |
| `TQ_B_MSE` | `b_mse` | 2 | PolarQuant bits/coord |
| `TQ_B_QJL` | `b_qjl` | 1 | QJL bits/coord |
| `TQ_DEVICE` | `device` | cuda | Torch device |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (standalone — no vLLM required)
python -m pytest tests/ -v

# Type check
mypy vllm_plugin/

# Lint
ruff check vllm_plugin/
```

The plugin uses `try/except` for vLLM imports, so tests can run without vLLM installed using mock stubs.

---

## Known Limitations

- **Triton kernels not yet integrated:** Current implementation uses pure PyTorch. Triton kernels for FWHT and fused dequant+attention are planned.
- **Accuracy in progress:** Cosine similarity is being debugged (S-matrix transpose bug identified). See `PLAN.md` Phase 1.
- **Paged cache bypassed:** TurboQuant manages its own compressed storage; vLLM's paged KV cache allocation is not used for KV data.
- **Attention sinks:** First few tokens are not yet preserved in higher precision (planned).

---

## License

Apache-2.0
