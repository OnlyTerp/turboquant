# TurboQuant Integrations

> Concrete, copy-pasteable setup for the serving engines and frameworks TurboQuant plugs
> into. Status and PR links as of **2026-04-17**. See
> [LANDSCAPE_2026.md](LANDSCAPE_2026.md) for the broader ecosystem survey.

---

## Quick navigation

- [vLLM](#vllm) — two paths: upstream PRs and our bundled backend plugin
- [SGLang](#sglang) — PR #21419 adds `--kv-cache-dtype turboquant`
- [llama.cpp](#llamacpp) — quantized-KV DP4A flash attention as of b8779
- [NVIDIA KVPress](#nvidia-kvpress) — stack TurboQuant precision under KVPress eviction
- [LMCache](#lmcache) — distributed / cross-session KV reuse on TurboQuant-compressed tensors
- [MLX (Apple Silicon)](#mlx-apple-silicon) — pure-PyTorch path for M-series Macs
- [Transformers / HuggingFace](#transformers--huggingface) — direct PyTorch monkey-patch
- [Docker + Helm](#docker--helm) — ship TurboQuant-serving rigs

---

## vLLM

**Status (Apr 17, 2026):** two upstream PRs are in motion. This repo also ships a
self-contained backend plugin under [`vllm_plugin/`](vllm_plugin/) that works against
vLLM ≥ 0.4.0 without waiting for upstream merge.

| Path | Status | Use when |
|---|---|---|
| [vLLM PR #39890 — official grouped `turboquant_3bit` / `turboquant_4bit`](https://github.com/vllm-project/vllm/pull/39890) | Open (2026-04-15) | You want first-class KV-cache-dtype support and are willing to track a PR branch |
| [vLLM PR #38662 — initial TurboQuant attention backend](https://github.com/vllm-project/vllm/pull/38662) | Open (2026-03-31) | You want the full fused attention backend, not just a cache dtype |
| [vLLM PR #39008 — legacy `tq4`](https://github.com/vllm-project/vllm/pull/39008) | **Closed** | Superseded by #39890 — avoid |
| Our [`vllm_plugin/`](vllm_plugin/) backend plugin | Scaffold, works on vLLM ≥ 0.4.0 | You want a plugin that installs with `pip install -e .` and does not require patching vLLM |

### Option A — Our bundled plugin (no upstream patching)

```bash
git clone https://github.com/OnlyTerp/turboquant.git
cd turboquant
pip install -e ".[vllm]"

# Serve. The plugin is auto-discovered via the vllm.platform_plugins entry point.
vllm serve meta-llama/Llama-3.1-8B-Instruct --attention-backend turboquant
```

Environment-variable overrides (all prefixed `TQ_`):

```bash
# Precision knobs
export TQ_B_MSE=2              # PolarQuant bits per coordinate (default 2 → "3.5-bit" mode)
export TQ_B_QJL=1              # QJL bits per coordinate
export TQ_FLUSH_INTERVAL=128   # Raw buffer size before compression

# GQA / model geometry (vLLM usually populates these automatically)
export TQ_NUM_LAYERS=32
export TQ_NUM_HEADS=32
export TQ_NUM_KV_HEADS=8       # Llama-3 / Llama-4 GQA
export TQ_HEAD_DIM=128

export TQ_DEVICE=cuda
```

See [`vllm_plugin/README.md`](vllm_plugin/README.md) for the Python API and architecture
details.

### Option B — Upstream PR #39890 (grouped official modes)

```bash
# Check out the PR branch
gh pr checkout 39890 --repo vllm-project/vllm

pip install -e . --no-build-isolation

# Run
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype turboquant_3bit \
  --max-model-len 131072
```

Also supports `turboquant_4bit` (higher quality, larger cache) and the legacy `*_nc`
presets for backward compatibility.

### Option C — Compose with NVFP4 weights on Blackwell

On SM100/SM120 (B100, B200, RTX 5090, RTX PRO 6000) the typical serving config is **NVFP4
weights × TurboQuant KV**:

```bash
# Weights: NVFP4 (requires pre-converted checkpoint)
# KV: TurboQuant 3.5-bit
vllm serve nvidia/Llama-3.1-8B-Instruct-NVFP4 \
  --quantization nvfp4 \
  --kv-cache-dtype turboquant_3bit
```

> As of Apr 17, 2026 the NVFP4-KV combination on desktop Blackwell (SM120) requires
> tracking open upstream issues in flashinfer and CUTLASS; see
> [Allen Kuo's Apr 16 writeup](https://allenkuo.medium.com/finishing-what-we-started-gemma-4-nvfp4-on-vllm-desktop-blackwell-wsl2-b2088c34815a)
> for the current workarounds on RTX PRO 6000 + WSL2.

---

## SGLang

**Status (Apr 17, 2026):** first-class `--kv-cache-dtype turboquant` via
[PR #21419](https://github.com/sgl-project/sglang/pull/21419). Ships with Triton kernels
for FWHT rotation, quantize/dequantize, and a fused 4-bit dequant kernel.

```bash
gh pr checkout 21419 --repo sgl-project/sglang
pip install -e "python[all]"

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype turboquant \
  --host 0.0.0.0 --port 30000
```

For Blackwell FP4 serving, also watch
[SGLang PR #21954](https://github.com/sgl-project/sglang/pull/21954) (NVFP4 KV cache,
SM100/SM120) — TurboQuant and NVFP4 KV will eventually be composable via the same
strategy-pattern abstraction.

---

## llama.cpp

**Status (Apr 17, 2026):** no TurboQuant-specific path upstream yet, but
[b8779 (Apr 13, 2026)](https://github.com/ggml-org/llama.cpp/releases/tag/b8779) shipped a
Vulkan **DP4A shader for quantized KV flash attention**, making quantized-KV decoding
fast enough to be the default on recent llama.cpp builds.

The closest off-the-shelf approximation to TurboQuant 3.5-bit today:

```bash
# Quantize both K and V caches (q4_0 is the widely-supported option)
./llama-cli \
  -m models/llama-3.1-8b-instruct.Q4_K_M.gguf \
  -ctk q4_0 -ctv q4_0 \
  -fa \
  -c 131072 \
  -p "your prompt"
```

Flags:

- `-ctk` / `-ctv` — cache type for K and V respectively. Options include `q4_0`, `q4_1`,
  `q5_0`, `q5_1`, `q8_0`, `iq4_nl`, `fp16`.
- `-fa` — flash attention, which in b8779+ has the DP4A path for quantized KV.
- `-c` — context length.

For a native TurboQuant path, watch the project
[github.com/hackimov/turboquant-kv](https://github.com/hackimov/turboquant-kv) — the
sibling port has been prototyping a ggml integration, though it is not yet upstream.

---

## NVIDIA KVPress

**Status (Apr 17, 2026):** KVPress is a framework of "press" strategies (ExpectedAttention,
ThinK, StreamingLLM, SnapKV, PyramidKV, ChunkPress, AdaKV). Use it for **token selection**
and layer TurboQuant underneath as the **precision backend**.

```bash
pip install kvpress==0.4.0
pip install -e /path/to/turboquant
```

```python
import torch
from kvpress import ExpectedAttentionPress
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuantCache, TurboQuantConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                             device_map="auto")

# 1. KVPress decides which tokens survive
press = ExpectedAttentionPress(compression_ratio=0.5)

# 2. TurboQuant compresses every surviving token
tq_config = TurboQuantConfig(
    num_layers=model.config.num_hidden_layers,
    num_heads=model.config.num_attention_heads,
    num_kv_heads=model.config.num_key_value_heads,
    head_dim=model.config.hidden_size // model.config.num_attention_heads,
    b_mse=2, b_qjl=1,
)

# Apply press hook that, after eviction, routes surviving tokens through
# TurboQuantCache.encode() / .decode().
with press(model, backend=TurboQuantCache(tq_config)):
    out = model.generate(**tokenizer(long_prompt, return_tensors="pt").to("cuda"),
                         max_new_tokens=256)
```

The combination "**KVPress eviction × TurboQuant precision**" is what the
[Apr 9 MarkTechPost guide](https://www.marktechpost.com/2026/04/09/an-end-to-end-coding-guide-to-nvidia-kvpress-for-long-context-llm-inference-kv-cache-compression-and-memory-efficient-generation/)
foreshadows — KVPress handles the budget, TurboQuant handles the bit width.

---

## LMCache

**Status (Apr 17, 2026):** [LMCache](https://github.com/LMCache/LMCache) is a distributed
KV cache that shares tensors across processes, nodes, and disk tiers. Their Apr 15 blog
post is the best layperson explainer of TurboQuant:
[blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/)

TurboQuant is a drop-in **tensor format** for LMCache:

```python
from lmcache import LMCache
from turboquant import TurboQuantCache, TurboQuantConfig

# Compose: LMCache for cross-process sharing, TurboQuant for per-token compression
lmcache = LMCache(backend="redis://cache.internal:6379")
tq_cache = TurboQuantCache(TurboQuantConfig(
    num_layers=32, num_heads=32, num_kv_heads=8, head_dim=128, b_mse=2, b_qjl=1))

# Store TurboQuant-compressed K/V under a document ID
lmcache.put(f"doc:{doc_id}", tq_cache.encode(k, v))

# Retrieve and decode (in the next session, or on another node)
k_hat, v_hat = tq_cache.decode(lmcache.get(f"doc:{doc_id}"))
```

The result: up to 4.9× smaller cache on disk + cross-session reuse. Pairs very well with
**KV Packet** ([arXiv 2604.13226](https://arxiv.org/html/2604.13226v1), Apr 14, 2026) for
context-independent document caching.

---

## MLX (Apple Silicon)

**Status (Apr 17, 2026):** MLX has no native FP4 tensor cores, but the TurboQuant
pure-PyTorch path runs on M-series Macs via the MPS backend. Quality matches the CUDA
path; throughput is CPU/GPU-bandwidth bound.

```bash
pip install -e .   # installs the pure-PyTorch TurboQuant
python src/demo.py
```

For weight-side INT4 on Apple Silicon, [ParoQuant](https://z-lab.ai/projects/paroquant/)
ships an `mlx` extra:

```bash
pip install "paroquant[mlx]"
python -m paroquant.cli.chat --model z-lab/Qwen3.5-4B-PARO
```

Then stack TurboQuant on top of the ParoQuant-quantized model by routing K/V through
`turboquant.TurboQuantCache` in the generation loop.

---

## Transformers / HuggingFace

For experimentation without a serving engine, you can monkey-patch the attention module:

```python
import torch
from transformers import AutoModelForCausalLM
from turboquant import TurboQuantCache, TurboQuantConfig

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16, device_map="auto",
)
cfg = TurboQuantConfig(
    num_layers=model.config.num_hidden_layers,
    num_heads=model.config.num_attention_heads,
    num_kv_heads=model.config.num_key_value_heads,
    head_dim=model.config.hidden_size // model.config.num_attention_heads,
)
tq_cache = TurboQuantCache(cfg)

# Replace the stock past_key_values with a TQ-backed cache object.
# (Full code in src/test_real_model.py.)
outputs = model.generate(
    **inputs,
    past_key_values=tq_cache,
    max_new_tokens=512,
)
```

See [`src/test_real_model.py`](src/test_real_model.py) for a full, runnable Mistral-7B
example.

---

## Docker + Helm

The [`deploy/`](deploy/) directory ships a reference Dockerfile:

```bash
docker build -t turboquant-vllm -f deploy/Dockerfile .

docker run --gpus all --ipc=host \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_ATTENTION_BACKEND=turboquant \
  -e TQ_B_MSE=2 -e TQ_B_QJL=1 \
  turboquant-vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 131072
```

For Kubernetes, the standard [vLLM Helm chart](https://github.com/vllm-project/production-stack)
works unchanged — just set the `extraEnv` block:

```yaml
extraEnv:
  - name: VLLM_ATTENTION_BACKEND
    value: turboquant
  - name: TQ_B_MSE
    value: "2"
  - name: TQ_B_QJL
    value: "1"
  - name: TQ_FLUSH_INTERVAL
    value: "128"
```

---

## Compatibility matrix

| Component | Version tested | Notes |
|---|---|---|
| Python | 3.10, 3.11, 3.12 | `match` statements used internally |
| PyTorch | 2.1+ | MPS works on Apple Silicon |
| CUDA | 11.8, 12.1, 12.4, 12.6 | 12.6+ recommended for Blackwell |
| Triton | 2.2+ (optional) | Enables GPU kernels in [`src/kernels.py`](src/kernels.py) |
| vLLM | 0.4.0+ | 0.9+ recommended for the plugin entry point |
| SGLang | main (PR #21419 branch) | Wait for merge for a stable tag |
| llama.cpp | b8779+ | DP4A flash-attn for quantized KV |
| KVPress | 0.4.0 | Stack eviction × precision |
| LMCache | latest | Cross-process cache sharing |

---

## Troubleshooting

### "Cache shapes don't match during decode"

You likely changed `b_mse` or `b_qjl` between encode and decode. The codebook is a
function of `(d, b_mse)`. If you persist caches across sessions (LMCache / KV Packet), pin
these values.

### "Accuracy degraded more than expected after switching to `rotation_mode="dense"`"

Verify you're using the same **seed** for encode and decode. Dense rotation is seeded
per-layer/head; mismatched seeds destroy correctness.

### "SGLang / vLLM plugin not registered"

Check that `setup.py` entry points are installed:

```bash
python -c "import importlib.metadata as im; \
  print([ep for ep in im.entry_points() if 'turboquant' in str(ep)])"
```

If empty, re-run `pip install -e .` with `--force-reinstall`.

### "Blackwell SM120 NVFP4 crashes with 'unsupported input dtype'"

As of Apr 17, 2026 the XQA MLA kernel on SM120 hard-codes FP8 for some paths
(see [flashinfer issue #2655](https://github.com/flashinfer-ai/flashinfer/issues/2655)).
Fall back to the Triton backend: `--attention-backend TRITON`.

### "Throughput is slow on H100"

The pure-PyTorch path is a reference, not a production kernel. On H100, use the Triton
kernels (`src/kernels.py`) or SGLang's fused kernels from PR #21419. The PyTorch path is
roughly 25× slower than FP16 attention; Triton closes ~80% of that gap.
