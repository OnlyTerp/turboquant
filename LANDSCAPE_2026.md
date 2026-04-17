# The 2026 KV-Cache Compression Landscape

> **Why this file exists.** TurboQuant is one entry in a fast-moving ecosystem. In the 10
> weeks around ICLR 2026 alone, at least a dozen new KV-cache compression methods shipped
> code or papers — many of them specifically benchmarked *against* TurboQuant. This guide
> orients you across the landscape so you can pick the right tool (or combine them) for
> your hardware, context length, and quality budget.
>
> Last refreshed: **2026-04-17** (Apr 14–17 = past 72 hours were especially busy — see
> [What's New](#whats-new-april-2026)).

---

## TL;DR — One-line positioning

| Method | Axis | Best at | TurboQuant relationship |
|---|---|---|---|
| **TurboQuant** (this repo) | **Rotation + scalar VQ + QJL residual** | **Data-oblivious, online, near information-theoretic optimal at 2.5–4 bpv** | — |
| **NVFP4 / MXFP4 KV** | Hardware-native FP4 block quant | Blackwell/SM100/SM120 throughput on large serving rigs | **Complementary** — TQ is the algorithm, NVFP4 is the container. They can coexist in one engine |
| **TriAttention** (Apr 2026) | Pre-RoPE Q/K concentration → trigonometric importance scoring | **Long reasoning (AIME-style 32K CoT)** via eviction, up to 10.7× KV memory reduction | **Orthogonal** — TQ compresses every retained token, TriAttention decides *which* tokens to retain |
| **Adaptive KV-quant** (Apr 6, 2026) | Learned per-token bit-width {2, 4, 8, FP16} | On-device / edge LLMs where static bits waste budget | Can wrap TQ as the 2-bit/4-bit backend |
| **LRKV (Low-Rank KV)** (Apr 9, 2026) | Shared full-rank basis + head-specific low-rank residual | Training-time architectural fix (45–53% KV reduction) | **Architectural** — applies at pretraining; TQ applies at inference |
| **DeepSeek MLA** | Multi-head Latent Attention: down-projection of K/V | Architectural compression, native to the model weights | Model-side; TQ still compresses the residual latent KV |
| **KIVI** (ICLR'24) | Per-channel K + per-token V asymmetric 2-bit | Simple, calibration-free 2-bit baseline | Superseded on LongBench by TQ at 3.5 bpv |
| **KVQuant** | Per-channel + pre-RoPE K + dense-sparse split | Sub-4-bit with outlier preservation | Outlier handling similar in spirit to TQ's two-rotation split |
| **ParoQuant** (ICLR'26, Apr 2026) | Scaled pairwise rotation for **weight** quant | INT4 **weights** with +2.4% over AWQ on reasoning | **Weight-side, not KV-side** — stack with TurboQuant for W4/KV3.5 |
| **SnapKV / H2O / PyramidKV / StreamingLLM** | Importance-based token eviction | High-ratio drop for long context (>100K) | **Orthogonal** — combine with TQ for multiplicative savings |
| **KV Packet** (Apr 14, 2026) | Context-independent cache reuse via soft-token adapters | Cross-document KV reuse in RAG without recompute | Orthogonal — caches TQ-compressed packets across sessions |
| **NVIDIA KVPress** | Framework of press strategies (ExpectedAttention, ThinK, etc.) | Plug-in eviction & budget control | Orthogonal — KVPress can drive *which* tokens TQ keeps |
| **[kvtc](https://github.com/OnlyTerp/kvtc)** (sibling project) | PCA-decorrelated channel rotation + per-channel scalar quant | Calibration-based KV compression when a few samples of real activations are available | **Sibling** — same authors, data-**aware** variant; TurboQuant stays data-*oblivious* for zero-setup deployments |

The short summary: **TurboQuant is a per-token precision compressor; the other axis is
token selection/eviction; the third axis is hardware format (FP4/FP8/INT4).** The three
compose.

---

## What's new (April 2026)

### Past 72 hours (Apr 14–17, 2026)

- **Apr 15 — [LMCache blog: "What is TurboQuant and why it matters for LLM inference"](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/)**
  by Kuntai Du (Tensormesh). Notes that TurboQuant is "really hot these days" and that
  some X/Twitter commentators are calling it "the most significant AI breakthrough this
  year." Frames TurboQuant as **the** answer to Jensen Huang's repeated GTC 2026 warnings
  about KV cache memory dominating inference cost.
- **Apr 15 — [vLLM PR #39890: "Add official 3-bit and 4-bit grouped TurboQuant modes"](https://github.com/vllm-project/vllm/pull/39890)**
  by @erhan1209. Adds canonical `turboquant_4bit` / `turboquant_3bit` KV-cache dtypes
  with grouped Triton store/decode paths, moving beyond the earlier `*_nc` legacy presets.
  Big step toward first-class TurboQuant in mainstream vLLM.
- **Apr 15 — [Towards AI hands-on: "Running a 35B Model Locally with TurboQuant"](https://pub.towardsai.net/running-a-35b-model-locally-with-turboquant-whats-actually-possible-right-now-1ac5327430b0)**
  Walks through the W4 weights × TQ3.5 KV stack on a single consumer GPU. Important
  clarification: **TurboQuant does not quantize weights** — it stacks *on top of* GGUF /
  AWQ / NVFP4.
- **Apr 14 — [arXiv 2604.13226 "KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs"](https://arxiv.org/html/2604.13226v1)**
  Trainable soft-token adapters that make cached documents reusable across contexts without
  recomputation. A nice partner to on-the-fly compression.
- **Apr 14 — [SGLang PR #22717: flashinfer TRTLLM backend for diffusion NVFP4](https://github.com/sgl-project/sglang/pull/22717)**
  More Blackwell/SM120 FP4 plumbing; same engineering thread as KV-side NVFP4.
- **Apr 14 — [NVFP4 KV cache PR #21954 in SGLang updated](https://github.com/sgl-project/sglang/pull/21954)**
  Part 1 of 4 for NVFP4 KV on SM120. Defines a `FP4KVCacheQuantMethod` strategy
  abstraction with `NVFP4KVMethod` (two-level scaling, per-tensor FP32 + per-block FP8
  E4M3) and `BlockFP4KVMethod` (MXFP4-like). Decode path reads raw FP4 buffers via XQA.

### Earlier April 2026

- **Apr 11 — ["TriAttention Compresses KV Cache 10.7×"](https://danilchenko.dev/posts/2026-04-11-triattention-kv-cache-compression-long-reasoning/)**
  Explanation of [arXiv 2604.04921](https://arxiv.org/abs/2604.04921) (Mao et al., Apr 6).
  Observes that pre-RoPE Q/K cluster around fixed centers; uses this "Q/K concentration"
  to score key importance via a trigonometric series. On **AIME25 with 32K-token
  generation**, TriAttention matches Full Attention accuracy at 2.5× higher throughput
  or 10.7× KV memory reduction — roughly double the accuracy of SnapKV / PyramidKV /
  StreamingLLM at the same ratio.
- **Apr 9 — [Fin.AI: "Low-Rank Key Value Attention: Reducing KV Cache Memory and Maintaining Head Diversity"](https://fin.ai/research/low-rank-key-value-attention-reducing-kv-cache-memory-and-maintaining-head-diversity/)**
  LRKV: drop-in MHA replacement. Shared full-rank KV basis plus head-specific low-rank
  residuals. 45–53% KV reduction vs standard MHA with **lower** test loss at model
  scales from 128M to 6.3B. Architectural (pretraining-time) fix, complementary to TQ.
- **Apr 9 — [MarkTechPost: "NVIDIA KVPress end-to-end guide"](https://www.marktechpost.com/2026/04/09/an-end-to-end-coding-guide-to-nvidia-kvpress-for-long-context-llm-inference-kv-cache-compression-and-memory-efficient-generation/)**
  Walk-through of [github.com/NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) — a
  framework that wraps multiple "press" strategies (ExpectedAttention, ThinK, StreamingLLM,
  etc.) behind one HuggingFace-compatible API.
- **Apr 6 — [arXiv 2604.04722 "Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs"](https://arxiv.org/abs/2604.04722)**
  Clemson group. Learned controller selects per-token precision from {2, 4, 8, FP16}
  based on token frequency, attention variance, and entropy. Beats static KV quant on
  SmolLM-135M/360M/1.7B benchmarks.
- **Apr 5 — [vLLM PR #39008 (closed): TurboQuant 4-bit (`tq4`) KV cache quantization](https://github.com/vllm-project/vllm/pull/39008)**
  Early community attempt at `--kv-cache-dtype tq4` with 2×int4 nibble packing and
  rotation pre-processing. Closed in favor of the grouped #39890 approach.
- **Apr 3 — [vLLM issue #38930 "Entropy-adaptive per-head KV cache quantization: +8% quality over uniform at same compression"](https://github.com/vllm-project/vllm/issues/38930)**
  Proposes per-head adaptive bit-widths selected from attention entropy, reports +8% vs
  uniform quantization at matched compression.
- **Apr 2 — [SGLang PR #21954: NVFP4 KV cache (1/4)](https://github.com/sgl-project/sglang/pull/21954)** — see above.

### March 2026 foundations (still very relevant)

- **Mar 25 — [SGLang PR #21419: TurboQuant KV cache compression (3–4 bit, ICLR 2026)](https://github.com/sgl-project/sglang/pull/21419)**
  First-class `--kv-cache-dtype turboquant` in SGLang. Triton kernels for FWHT rotation,
  quantize/dequantize, bit-packing, and a fused 4-bit dequant kernel.
- **Mar 16 — [vLLM PR #37192 (WIP, closed): KVCACHE NVFP4](https://github.com/vllm-project/vllm/pull/37192)** — Triton backend, FP4 E2M1 two-level scaling, ~3.5× over BF16 at head_size=128.
- **Mar 13 — [FlashAttention-4 paper release](https://arxiviq.substack.com/p/flashattention-4-algorithm-and-kernel)**
  Blackwell-first attention kernel. Tensor core throughput ≈ 2.25 PFLOPS FP16/BF16 vs
  H100's 1 PFLOPS. Native FP4/FP8 paths relevant for quantized KV decode.
- **Mar 8 — [FlashAttention-4 on Blackwell overview](https://medium.com/ai-in-plain-english/flashattention-4-supercharging-transformer-attention-on-nvidia-blackwell-gpus-cf81caa64b48)**
- **Feb 24 — [DeepSeek-V3 MLA in JAX: memory-complexity analysis](https://building.theatlantic.com/deepseek-v3-mla-vs-mha-a-jax-native-benchmark-of-inference-efficiency-1854677efc03)**
  Reminder that MLA (down-projection of K/V into a shared latent) is the architectural
  baseline to beat for KV compression — DeepSeek-V3 ships it natively.

### The model-side pressure

These releases are why KV compression is suddenly a P0 infrastructure concern:

| Model (2026) | Context | Why it matters |
|---|---|---|
| **Llama 4 Scout** | 10M tokens (!) | [Needs a 4× H100 cluster to run FP16](https://agentscookbook.com/docs/compare/llama-4-vs-qwen-3-5/); KV dominates memory |
| **Llama 4 Maverick** | 1M tokens, 402B MoE (17B active) | Enterprise 7× H200 — KV compression turns this into 1× H200 territory |
| **Qwen 3.5 397B** | 256K tokens | Frontier-adjacent quality in open weights; KV compression makes consumer on-prem plausible |
| **Qwen 3.5 9B** | 256K tokens | Runs on a gaming desktop or 16GB Apple Silicon — TQ pushes context further |
| **Gemma 4 31B / 4 family** | Various | [NVFP4 Blackwell benchmark Apr 2–16, 2026](https://www.millstoneai.com/inference-benchmark/gemma-4-31b-nvfp4-1x-rtx-pro-6000-blackwell) |
| **OpenClaw 32B** | Long reasoning | The model TriAttention explicitly targets for single-4090 deployment |

---

## The three compression axes

Think of KV-cache compression as three orthogonal knobs:

```
                  PRECISION (bits/value)
                          │
                          │
                          │
                     TurboQuant
                     NVFP4, FP8
                     KIVI, KVQuant
                          │
     TOKEN SELECTION ─────┼───── CONTAINER / LAYOUT
       H2O, SnapKV              (FP4 blocks, INT4 pack,
       TriAttention             grouped scales, XQA kernels)
       PyramidKV, KVPress
```

You can (and should) combine axes:

```
Weights:  AWQ / GPTQ / NVFP4 / MXFP4  (4-bit)
  ×
KV precision:  TurboQuant 3.5-bit  (near-FP16 quality)
  ×
Token selection:  TriAttention / SnapKV  (10× on long context)
  ×
Container:  NVFP4-backed cache on Blackwell  (hardware speedup)
```

The compounding is why serving a Llama-4-Maverick-class model on one workstation is
suddenly plausible in 2026 — no single axis gets you there.

---

## Method-by-method deep dive

### TurboQuant (this repo)

- **Algorithm:** random rotation (Hadamard or Haar) → scalar Lloyd-Max per coordinate →
  QJL 1-bit residual for unbiased inner product.
- **Strengths:** **data-oblivious**, no calibration, zero compounding across layers
  (scalar per-coordinate, no coupling), provably within 2.7× of information-theoretic
  optimal distortion.
- **Benchmarks:** 3.5 bpv = identical LongBench on Llama-3.1-8B; 2.5 bpv = marginal
  degradation (from the paper).
- **Weaknesses:** pure PyTorch reference is ~25× slower than FP16 on CPU; Triton kernels
  experimental; outlier handling differs slightly from paper (see
  [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)).

### TriAttention (arXiv 2604.04921, Apr 2026)

- **Insight:** all existing eviction methods (H2O, SnapKV, PyramidKV, StreamingLLM) score
  key importance using **post-RoPE** queries. But RoPE rotates queries per position, so a
  recent query gives a misleading "viewpoint" for older keys. TriAttention goes to
  **pre-RoPE** space, where Q and K concentrate around fixed non-zero centers, and uses a
  **trigonometric series** over position deltas to score keys.
- **Result:** on AIME25 32K CoT, **10.7× KV reduction** or **2.5× throughput** with zero
  accuracy loss. Baselines at matched compression get roughly half the accuracy.
- **Relationship to TurboQuant:** purely orthogonal — TriAttention decides which tokens
  to keep, TurboQuant compresses each kept token. Stack them.
- Code: [github.com/WeianMao/triattention](https://github.com/WeianMao/triattention).

### Adaptive KV-Quant (arXiv 2604.04722, Apr 2026)

- **Insight:** Huffman-style variable-length bit allocation. A tiny controller uses
  token features (frequency, attention variance, entropy-based uncertainty) to pick
  precision from {2, 4, 8, FP16} at decode time.
- **Target:** on-device LLMs (SmolLM-135M/360M/1.7B).
- **Relationship to TurboQuant:** could use TQ2.5 / TQ3.5 / FP16 as the three tiers of a
  learned policy. The controller picks the tier; TQ runs the tier.

### LRKV — Low-Rank Key-Value attention (Fin.AI, Apr 2026)

- **Insight:** heads aren't independent. MHA is too expensive, MQA/GQA is too
  restrictive. LRKV uses a **shared full-rank KV basis plus head-specific low-rank
  residuals** — a continuous knob between MQA and MHA.
- **Result:** 45–53% KV reduction vs MHA with **lower test loss** across 128M → 6.3B
  model scales; stronger downstream performance after SFT.
- **Relationship:** architectural (pretraining time). TurboQuant still compresses the
  LRKV latent at inference. They multiply.

### DeepSeek-V3 Multi-head Latent Attention (MLA)

- K/V down-projected into a shared latent; each head up-projects at use time. Becomes
  the attention baseline to beat — DeepSeek-V3 ships it in production.
- TurboQuant applies to the latent vectors in MLA the same way it does to K/V vectors in
  MHA — so these compose naturally.

### NVFP4 / MXFP4 KV (Blackwell, 2026)

- **NVFP4:** 4-bit E2M1 values with **two-level scaling** (per-tensor FP32 + per-block
  FP8 E4M3). ~3.5× compression over BF16 at head_size=128. Native Blackwell tensor cores.
- **MXFP4:** OCP Microscaling standard — single-level scaling with 32-element blocks.
- **Where to find it:**
  - [SGLang PR #21954](https://github.com/sgl-project/sglang/pull/21954) (XQA attention
    backend, SM100/SM120)
  - [vLLM PR #37192](https://github.com/vllm-project/vllm/pull/37192) (closed, but the
    Triton backend code is useful reference)
- **Relationship:** NVFP4 is a **container**. TurboQuant's 3.5-bit quantized index can
  live inside NVFP4 blocks on Blackwell. Think of NVFP4 as how the GPU stores the data
  and TurboQuant as what the data represents.

### NVIDIA KVPress

- Not a single method — a **framework** exposing many "press" strategies behind one API.
- Includes ExpectedAttention, ThinK, StreamingLLM, SnapKV, PyramidKV, ChunkPress, AdaKV.
- `pip install kvpress`; drop-in HuggingFace generation.
- Link: [github.com/NVIDIA/kvpress](https://github.com/NVIDIA/kvpress).
- **Relationship:** KVPress chooses which tokens to keep; TurboQuant compresses each
  retained token. Use KVPress to drive eviction, TurboQuant as the precision backend.

### KIVI (ICLR'24) — still the obvious 2-bit baseline

- Per-**channel** quantization for K (because K has channel-wise outliers post-RoPE) +
  per-**token** for V (outlier-free). 2.25 bpv.
- Simpler and earlier than TQ; TQ's 3.5-bit beats KIVI-2bit on LongBench by ~1.5 pts on
  Llama-3.1-8B (paper Table 1).

### KVQuant

- Per-channel + pre-RoPE K quantization; dense-and-sparse split for outliers; non-uniform
  codebooks. Sub-4-bit with small perplexity hit.
- Philosophically close to TQ's two-independent-rotations outlier handling, but
  calibration-based rather than data-oblivious.

### ParoQuant (ICLR'26, Apr 2026)

- **Weight-side** INT4 quantization using **scaled pairwise rotation**. +2.4% over AWQ on
  reasoning benchmarks.
- [z-lab.ai/projects/paroquant](https://z-lab.ai/projects/paroquant/).
- **Relationship:** complementary — quantize weights with ParoQuant, KV with TurboQuant.

### Eviction family: H2O, SnapKV, PyramidKV, StreamingLLM, Keyformer

- All orthogonal to TurboQuant. Use them to decide **which** tokens survive; use
  TurboQuant to decide **how precisely** to store the survivors.
- TriAttention is the 2026 successor most serious about long-reasoning workloads.

### KV Packet (arXiv 2604.13226, Apr 14, 2026)

- Trainable soft-token adapters wrap cached documents so they can be reused across
  contexts without recomputation. Addresses RAG-style reuse (cf. CacheBlend, EPIC,
  SAM-KV). Near-zero FLOPs and lower TTFT than recomputation baselines.
- **Relationship:** TurboQuant-compressed "packets" can be stored and reused via KV
  Packet adapters.

### LMCache

- Distributed KV cache with cross-process sharing and disk offload.
- [github.com/LMCache/LMCache](https://github.com/LMCache/LMCache)
- [The Apr 15 LMCache blog post](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/)
  is the best layperson explainer of *why* TurboQuant matters and is what kicked off
  much of the current X/Twitter buzz.

---

## Decision guide: which compressor do I actually want?

> Use this to pick a starting configuration. All of these can be **stacked** — this chart
> just tells you where to start.

| Scenario | Start with | Then add | Why |
|---|---|---|---|
| **Datacenter Blackwell (B100/B200), max throughput** | NVFP4 weights + NVFP4 KV | — | Hardware-native FP4 path is the throughput king |
| **Datacenter Hopper (H100/H200), long context (>128K)** | FP8 weights + **TurboQuant 3.5-bit KV** | SnapKV / KVPress if you're beyond 1M tokens | TQ beats KIVI-2bit quality at matched memory; H100 FP8 is native |
| **Consumer Blackwell (RTX 5090, RTX PRO 6000 96GB)** | NVFP4 weights + **TurboQuant 3.5-bit KV** | — | RTX PRO 6000 96GB + TQ = Llama-4-Maverick-class on one card |
| **Consumer Ada (RTX 4090/4080)** | AWQ-INT4 weights + **TurboQuant 3.5-bit KV** | TriAttention / SnapKV for 32K+ reasoning | Matches the [TriAttention single-4090 OpenClaw result](https://arxiv.org/abs/2604.04921) |
| **Apple Silicon (M3/M4/M5)** | MLX-INT4 weights + TurboQuant (CPU path) | — | MLX lacks NVFP4; TQ pure-PyTorch path works |
| **On-device / edge (phone, Jetson)** | Adaptive KV-Quant or TurboQuant-2.5-bit | Token eviction (H2O / StreamingLLM) | Per-token bit allocation is essential at these budgets |
| **RAG systems, high cache reuse** | TurboQuant 3.5-bit KV | **KV Packet** or **LMCache** | Stack across-session reuse on top of per-token compression |
| **Reasoning / long CoT (AIME, AIME25, math)** | TurboQuant 3.5-bit KV | **TriAttention** | Token selection dominates at 32K+ CoT; TQ handles the precision |
| **Pretraining a new model** | **LRKV** attention or **MLA** | TurboQuant at inference time | Architectural fix + inference compression multiply |
| **I just want something that works today** | **TurboQuant 3.5-bit KV** (this repo) | — | Calibration-free, data-oblivious, 3 lines to integrate |

---

## Where TurboQuant fits in each serving engine

See [INTEGRATIONS.md](INTEGRATIONS.md) for concrete commands and links. Summary:

| Engine | Status (Apr 17, 2026) | How to use TurboQuant |
|---|---|---|
| **vLLM** | WIP (PRs [#38662](https://github.com/vllm-project/vllm/pull/38662), [#39890](https://github.com/vllm-project/vllm/pull/39890)) | `--kv-cache-dtype turboquant_3bit` (PR #39890); or our `vllm_plugin/` backend plugin |
| **SGLang** | WIP (PR [#21419](https://github.com/sgl-project/sglang/pull/21419)) | `--kv-cache-dtype turboquant` |
| **llama.cpp** | DP4A flash-attn for quantized KV in [b8779 (Apr 13)](https://github.com/ggml-org/llama.cpp/releases/tag/b8779); TQ-specific path not yet upstream | Use `-ctk q4_0 -ctv q4_0` today for a close proxy |
| **LMCache** | First-class coverage in [their Apr 15 blog post](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/) | Compose TQ-compressed tensors inside LMCache cross-session cache |
| **NVIDIA KVPress** | Eviction/selection framework | Use KVPress for token eviction, TQ for per-token precision |
| **MLX (Apple)** | No native FP4; use TQ PyTorch path | `pip install -e .` and run `src/demo.py` on Apple Silicon |

---

## Cross-method accuracy-at-compression (LongBench Avg, Llama-3.1-8B)

Normalized to the TurboQuant paper's Table 1 plus independent reports.

| Method | KV bits | LongBench Avg | Needle @128K | Notes |
|---|---|---|---|---|
| Full precision (FP16) | 16 | 50.06 | 0.997 | baseline |
| **TurboQuant (3.5-bit)** | 3.5 | **50.06** | **0.997** | near-zero loss |
| PolarQuant (predecessor) | 3.9 | 49.78 | 0.995 | |
| **TurboQuant (2.5-bit)** | 2.5 | 49.44 | 0.997 | marginal degradation |
| KIVI | 3 | 48.50 | 0.981 | channel-K / token-V |
| SnapKV (no precision) | 16 at reduced budget | 44.57 | 0.858 | eviction only |
| StreamingLLM | — | 41.2 (reported) | 0.72 | eviction only |
| H2O | — | 42.8 (reported) | 0.79 | eviction only |

For **reasoning at 32K CoT** (AIME25), the picture flips — TriAttention's eviction
becomes dominant, and precision methods alone under-perform. See the
[TriAttention paper](https://arxiv.org/abs/2604.04921) Fig. 1 for the trade-off curve.

---

## Further reading

- **Original paper:** [TurboQuant: Online Vector Quantization with Near-optimal Distortion
  Rate](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni; ICLR 2026).
- **ICLR 2026 poster:** [iclr.cc/virtual/2026/poster/10006985](https://iclr.cc/virtual/2026/poster/10006985)
  (Sat, Apr 25, 2026, 11:15 AM PDT).
- **Companion papers:**
  - [QJL (AAAI 2025)](https://dl.acm.org/doi/10.1609/aaai.v39i24.34773) — the 1-bit
    sign-based JL transform TurboQuant uses for its residual.
  - [PolarQuant (arXiv 2502.02617)](https://arxiv.org/abs/2502.02617) — the recursive
    polar-coordinate predecessor TurboQuant deliberately avoids.
- **Related implementations:**
  - [hackimov/turboquant-kv](https://github.com/hackimov/turboquant-kv) — sibling
    open-source port with alternative kernel design.
- **Our own:**
  - [README.md](README.md) — quickstart and headline numbers.
  - [BENCHMARKS.md](BENCHMARKS.md) — memory tables, throughput targets, and methodology.
  - [INTEGRATIONS.md](INTEGRATIONS.md) — vLLM / SGLang / llama.cpp / MLX / KVPress
    concrete commands.
  - [FAQ.md](FAQ.md) — common misconceptions ("does this replace AWQ?", etc.).
  - [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) — rotation modes, outlier
    channels, QJL projection, Blackwell stacking.
  - [pseudocode.md](pseudocode.md) — line-by-line paper pseudocode for re-implementers.
