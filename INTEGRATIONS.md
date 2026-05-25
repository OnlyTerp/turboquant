# TurboQuant Integrations

> Concrete, copy-pasteable setup for the serving engines and frameworks TurboQuant plugs
> into. Status and PR links as of **2026-05-25**. See
> [LANDSCAPE_2026.md](LANDSCAPE_2026.md) for the broader ecosystem survey.
>
> **Before you reach for TurboQuant, try `--kv-cache-dtype fp8` first.** The May 2026
> [Red Hat AI evaluation](https://blog.vllm.ai/2026/05/11/turboquant.html) showed FP8 KV
> matches BF16 quality on every benchmark with no throughput penalty. Only choose TurboQuant
> when you need more than 2× compression. See [FAQ "Should I use FP8 or TurboQuant?"](FAQ.md#should-i-use-fp8-or-turboquant).

---

## Quick navigation

- [vLLM](#vllm) — **merged upstream**; production fork available
- [SGLang](#sglang) — WIP, multiple competing PRs
- [llama.cpp](#llamacpp) — community forks for both head_dim=128 and head_dim=256
- [NVIDIA KVPress](#nvidia-kvpress) — stack TurboQuant precision under KVPress eviction
- [LMCache](#lmcache) — distributed / cross-session KV reuse on TurboQuant-compressed tensors
- [MLX (Apple Silicon)](#mlx-apple-silicon) — Metal kernel port from varjoranta
- [Transformers / HuggingFace](#transformers--huggingface) — direct PyTorch monkey-patch
- [Docker + Helm](#docker--helm) — ship TurboQuant-serving rigs

---

## vLLM

**Status (May 25, 2026):** TurboQuant is **merged upstream** in vLLM 0.20.x. You can
also install a production fork with CUDA dequant kernels, or use this repo's
self-contained plugin if you want a `pip install`-able path without patching vLLM.

| Path | Status | Use when |
|---|---|---|
| **Upstream `vllm>=0.20.2` `--kv-cache-dtype turboquant_*`** | **Merged** ([docs.vllm.ai](https://docs.vllm.ai/en/v0.20.2/api/vllm/model_executor/layers/quantization/turboquant/)) | **Default. Use this.** |
| [`turboquant-plus-vllm`](https://pypi.org/project/turboquant-plus-vllm/) production fork | v0.13.5 (May 13, 2026) | You need CUDA dequant kernels, MoE support, or native-packed TQ checkpoints |
| [0xSero/turboquant](https://github.com/0xSero/turboquant) Triton reference | 1.1K stars | Research / kernel development; benchmarked on RTX 5090 and 8×RTX 3090 |
| Our [`vllm_plugin/`](vllm_plugin/) backend plugin | Scaffold, works on vLLM ≥ 0.4.0 | You're stuck on an older vLLM and can't patch it |

### Option A — Upstream vLLM (recommended)

```bash
pip install "vllm>=0.20.2"

# Use the recommended TQ variant — 4-bit with norm correction (no QJL)
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --kv-cache-dtype turboquant_4bit_nc \
  --max-model-len 131072
```

Four variants ship in `vllm>=0.20.2`:

| Variant | Bits per value | Quality | When to use |
|---|---|---|---|
| `turboquant_k8v4` | 8 K / 4 V | Small drop vs FP8 | Almost always dominated by `fp8` — rarely the right pick |
| `turboquant_4bit_nc` | 4 (no QJL) | Within ~1 pt of FP8 | **The recommended TQ config** |
| `turboquant_k3v4_nc` | 3 K / 4 V (no QJL) | 10–15 pt drop on reasoning | Qwen-class K-dominant models with extreme K/V norm asymmetry |
| `turboquant_3bit_nc` | 3 (no QJL) | 15–25 pt drop on reasoning at long context | Edge / strict memory budget only |

### Option B — `turboquant-plus-vllm` (production fork with CUDA kernels)

Use this when you want the 10.1× decode speedup on Qwen3.6-35B-A3B, or when you need
the `varjosoft/Qwen3.6-35B-A3B-TQ3-native` checkpoint (the first frontier MoE model with
weights natively packed in TQ format).

```bash
pip install turboquant-plus-vllm   # bundles vLLM 0.20.2 + FlashInfer 0.5

vllm serve varjosoft/Qwen3.6-35B-A3B-TQ3-native \
  --kv-cache-dtype turboquant_3bit_native \
  --max-model-len 131072
```

The `*_native` dtypes assume the model checkpoint already has its weights rotated into
TQ-friendly layout; the per-expert rotation is baked into the weight tensor, which is
what lets the CUDA kernel hit ~49 t/s on RTX 5090 with Qwen3.6-35B-A3B.

### Option C — Our bundled plugin (no upstream patching)

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
export TQ_B_QJL=0              # QJL bits per coordinate (recommend 0 — May 2026 finding)
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

### Option D — Compose with NVFP4 weights on Blackwell

On SM100/SM120 (B100, B200, RTX 5090, RTX PRO 6000) the typical serving config is **NVFP4
weights × FP8 KV**. Switch the KV to TurboQuant only if FP8 doesn't give you enough
headroom:

```bash
# Default Blackwell config: NVFP4 weights + FP8 KV
vllm serve nvidia/Llama-3.3-70B-Instruct-NVFP4 \
  --quantization nvfp4 \
  --kv-cache-dtype fp8

# Switch to TurboQuant only if FP8 KV doesn't fit
vllm serve nvidia/Llama-3.3-70B-Instruct-NVFP4 \
  --quantization nvfp4 \
  --kv-cache-dtype turboquant_4bit_nc
```

> As of May 2026 the NVFP4-KV combination on desktop Blackwell (SM120) still requires
> tracking open upstream issues in flashinfer and CUTLASS; see
> [Allen Kuo's Apr 16 writeup](https://allenkuo.medium.com/finishing-what-we-started-gemma-4-nvfp4-on-vllm-desktop-blackwell-wsl2-b2088c34815a)
> for current workarounds on RTX PRO 6000 + WSL2.

---

## SGLang

**Status (May 25, 2026):** SGLang has **three competing PRs**, none merged. Watch this
space — the SGLang maintainers have stated they will pick one approach once the design
shakes out. For production today, use vLLM instead.

| PR | Approach | Last activity |
|---|---|---|
| [#21419](https://github.com/sgl-project/sglang/pull/21419) | First-class `--kv-cache-dtype turboquant` with Triton kernels for FWHT + dequant | Mar 30, 2026 — stale |
| [#22048](https://github.com/sgl-project/sglang/pull/22048) | Cleaner abstraction, integrates with SGLang's strategy pattern | May 4, 2026 — active |
| [#23135](https://github.com/sgl-project/sglang/pull/23135) | Triton kernels only (no integration), waiting for the strategy PR | May 19, 2026 — active |

If you want to experiment today, the most-active branch is #22048:

```bash
gh pr checkout 22048 --repo sgl-project/sglang
pip install -e "python[all]"

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.3-70B-Instruct \
  --kv-cache-dtype turboquant \
  --host 0.0.0.0 --port 30000
```

For Blackwell FP4 serving, also watch
[SGLang PR #21954](https://github.com/sgl-project/sglang/pull/21954) (NVFP4 KV cache,
SM100/SM120) — TurboQuant and NVFP4 KV will eventually be composable via the same
strategy-pattern abstraction.

---

## llama.cpp

**Status (May 25, 2026):** Two community forks support native TurboQuant in llama.cpp.
The choice between them is driven by your model's head_dim:

| Fork | Head dim support | Best at |
|---|---|---|
| [**spiritbuun/llama-cpp-turboquant-cuda**](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) | 128 only (Llama, Qwen2.5, Qwen3 ≤ 14B, Mistral) | Fast on Llama-3 family, smaller binary |
| [**AmesianX/TurboQuant**](https://github.com/AmesianX/TurboQuant) | **Any** (128, 192, 256, 384) | Required for Qwen3.5/3.6, Gemma 4 (head_dim=256), DeepSeek-V3 (head_dim=192) |

The AmesianX fork is the safer default — its P1→P5 head_dim detection cascade gracefully
handles models where `n_embd/n_head` lies (Qwen3.5, Qwen3.6, Gemma 4), and as of
[v1.4.2](https://github.com/AmesianX/TurboQuant/releases/tag/v1.4.2) (Apr 25, 2026) it
has MMA tensor-core acceleration on the inverse-WHT dequant path (49 t/s on RTX 5090 +
Qwen3.6-35B-A3B).

```bash
# Build the AmesianX fork with CUDA
git clone https://github.com/AmesianX/TurboQuant
cd TurboQuant
make GGML_CUDA=1 -j

# Serve. Use tq3/tq4 cache types (TurboQuant-native) or fall back to q4_0 for compat
./llama-cli \
  -m models/qwen3.6-35b-a3b.gguf \
  -ctk tq3 -ctv tq3 \
  -fa \
  -c 131072 \
  -p "your prompt"
```

Flags:

- `-ctk` / `-ctv` — cache type. The TurboQuant-native types are `tq3` and `tq4`. Standard
  `q4_0` / `q4_1` / `q5_0` / `q5_1` / `q8_0` / `iq4_nl` / `fp16` are also supported as
  fallbacks.
- `-fa` — flash attention. In [b8779+](https://github.com/ggml-org/llama.cpp/releases/tag/b8779)
  upstream this uses a Vulkan DP4A shader for quantized KV; both forks override with their own CUDA path.
- `-c` — context length.

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
from turboquant import TurboQuantCache

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                             device_map="auto")

# 1. KVPress decides which tokens survive
press = ExpectedAttentionPress(compression_ratio=0.5)

# 2. TurboQuant compresses every surviving token. The public TurboQuantCache
#    takes positional (n_layers, n_heads, d=head_dim, b_mse=2, ...).
head_dim = model.config.hidden_size // model.config.num_attention_heads
tq_cache = TurboQuantCache(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,   # compress one cache per KV head (GQA)
    d=head_dim,
    b_mse=2,                                     # 2-bit MSE regular + 1-bit QJL residual
    mixed_precision=True,                         # outlier channels get b_mse+1 bits
    device=torch.device("cuda"),
)

# Apply press hook that, after eviction, routes surviving tokens through
# TurboQuantCache.store() / .attention_scores().
with press(model, backend=tq_cache):
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
from turboquant import TurboQuantCache

# Compose: LMCache for cross-process sharing, TurboQuant for per-token compression.
# The public TurboQuantCache takes positional (n_layers, n_heads, d, b_mse, ...).
lmcache = LMCache(backend="redis://cache.internal:6379")
tq_cache = TurboQuantCache(
    n_layers=32,
    n_heads=8,          # compress once per KV head under GQA
    d=128,
    b_mse=2,
    mixed_precision=True,
)

# Store TurboQuant-compressed K/V under a document ID. tq_cache.store() writes into
# per-(layer, head) ring buffers; for cross-session reuse you serialize the resulting
# TurboQuantCompressed tuples (q_codes, residual_signs, norms) — see src/cache.py.
for layer in range(tq_cache.n_layers):
    for head in range(tq_cache.n_heads):
        tq_cache.store(layer, head, k[layer, head], v[layer, head])
lmcache.put(f"doc:{doc_id}", tq_cache.cache)

# On retrieval, attach the compressed tuples back to a freshly-built TurboQuantCache
# with identical (n_layers, n_heads, d, b_mse) and call .attention_scores(q).
tq_cache.cache = lmcache.get(f"doc:{doc_id}")
```

The result: up to 4.9× smaller cache on disk + cross-session reuse. Pairs very well with
**KV Packet** ([arXiv 2604.13226](https://arxiv.org/html/2604.13226v1), Apr 14, 2026) for
context-independent document caching.

---

## MLX (Apple Silicon)

**Status (May 25, 2026):** MLX has no native FP4 tensor cores, but the
[varjoranta/turboquant-vllm MLX port](https://github.com/varjoranta/turboquant-vllm/tree/main/mlx)
adds native Metal kernels for the FWHT rotation and 3/4-bit dequant. This is faster
than the pure-PyTorch MPS path that this repo ships and is what the community converges
on for M-series Macs.

```bash
# Install the varjoranta MLX port
pip install "turboquant-mlx"   # PyPI package from the varjoranta repo

# Run a chat loop with TurboQuant KV compression on M3/M4/M5
python -m turboquant.mlx.chat \
  --model mlx-community/Llama-3.3-70B-Instruct-4bit \
  --kv-bits 4 \
  --kv-mode nc
```

If you want to stay on this repo's pure-PyTorch path (slower, but uses the public API):

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
from turboquant import TurboQuantCache

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16, device_map="auto",
)
head_dim = model.config.hidden_size // model.config.num_attention_heads
tq_cache = TurboQuantCache(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,   # GQA: compress per KV head
    d=head_dim,
    b_mse=2,
    mixed_precision=True,
    device=torch.device("cuda"),
)

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
  -e TQ_B_MSE=2 -e TQ_B_QJL=0 \
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
    value: "0"            # disable QJL per May 2026 community consensus
  - name: TQ_FLUSH_INTERVAL
    value: "128"
```

---

## Compatibility matrix

| Component | Version tested | Notes |
|---|---|---|
| Python | 3.10, 3.11, 3.12 | `match` statements used internally |
| PyTorch | 2.1+ | MPS works on Apple Silicon |
| CUDA | 11.8, 12.1, 12.4, 12.6, 12.8 | 12.8+ recommended for Blackwell SM120 |
| Triton | 2.2+ (optional) | Enables GPU kernels in [`src/kernels.py`](src/kernels.py) |
| vLLM (upstream) | **0.20.2+** | Merged `--kv-cache-dtype turboquant_*` |
| vLLM (production fork) | turboquant-plus-vllm 0.13.5 | Bundles CUDA dequant + FlashInfer CUTLASS MoE |
| vLLM (plugin path) | 0.4.0+ | 0.9+ recommended for the plugin entry point |
| SGLang | main — PRs [#21419](https://github.com/sgl-project/sglang/pull/21419), [#22048](https://github.com/sgl-project/sglang/pull/22048), [#23135](https://github.com/sgl-project/sglang/pull/23135) | None merged as of May 2026 |
| llama.cpp | AmesianX v1.4.2 / spiritbuun latest | Use AmesianX for head_dim ≥ 192 |
| KVPress | 0.4.0 | Stack eviction × precision |
| LMCache | latest | Cross-process cache sharing |

---

## Troubleshooting

### "Cache shapes don't match during decode"

You likely changed `b_mse` (or `b_outlier`) between encode and decode. The codebook is a
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

The pure-PyTorch path is a reference, not a production kernel. For production on H100:

1. **First, try `--kv-cache-dtype fp8`.** It matches BF16 quality at zero throughput
   cost on Hopper/Blackwell. If that fits your memory budget, you don't need TurboQuant.
2. If you need >2× compression, use
   [`turboquant-plus-vllm`](https://pypi.org/project/turboquant-plus-vllm/) (CUDA dequant
   kernels, 10.1× over the reference path) or
   [0xSero/turboquant](https://github.com/0xSero/turboquant) (Triton kernels).
3. The plain PyTorch path is roughly 25× slower than FP16 attention; the Triton kernels
   in `src/kernels.py` close ~80% of that gap but still aren't competitive with the
   production forks.

### "Output quality is worse than the paper claims"

The most common culprits, in order:

1. **You're using a QJL-augmented variant at b ≤ 3.** Switch to a `*_nc` (no-QJL) variant
   or set `qjl_score_weight=0.0`. See [FAQ "Does the QJL step actually help?"](FAQ.md#does-the-qjl-step-actually-help).
2. **You're not skipping the first / last 2 layers.** vLLM's `*_nc` paths do this
   automatically; custom integrations need to add it.
3. **Your model has head_dim > 128 and you're using the spiritbuun llama.cpp fork.**
   Switch to [AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant), which detects
   head_dim correctly for Qwen3.5, Qwen3.6, Gemma 4.
4. **You're using 3-bit modes on reasoning workloads.** The Red Hat eval showed 15–25
   pt drops on AIME25 at long context with `turboquant_3bit_nc`. Use `turboquant_4bit_nc`
   or fall back to FP8 KV.
5. **You're benchmarking with cosine similarity on random vectors instead of MRCR /
   LongBench / AIME25.** Cosine similarity doesn't predict end-to-end quality — see
   [FAQ "What's the right way to benchmark my model?"](FAQ.md#whats-the-right-way-to-benchmark-my-model).
