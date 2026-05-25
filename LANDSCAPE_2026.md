# The 2026 KV-Cache Compression Landscape

> **Why this file exists.** TurboQuant is one entry in a fast-moving ecosystem. In the
> 14 weeks between the ICLR 2026 poster and now, the community has gone from "TurboQuant
> is SOTA, use it everywhere" to a much more textured picture in which TurboQuant is the
> right tool only in a specific regime. This guide orients you across the landscape so
> you can pick the right tool (or combine them) for your hardware, context length, and
> quality budget.
>
> Last refreshed: **2026-05-25**. The biggest update vs the April 17 revision: the
> [Red Hat AI / vLLM evaluation (May 11)](https://blog.vllm.ai/2026/05/11/turboquant.html)
> and the community consensus around dropping QJL.

---

## The big May 2026 update

Three things changed since the previous refresh:

1. **The Red Hat AI / vLLM team published the first comprehensive third-party evaluation**
   of TurboQuant in [a May 11 blog post](https://blog.vllm.ai/2026/05/11/turboquant.html).
   Across Llama-3.3-70B, Qwen3-30B-A3B, and MiniMax-M2.7 on long-context retrieval (MRCR)
   and reasoning (AIME25, GPQA, MATH500, LiveCodeBench-v6), they conclude:
   - **FP8 (E4M3) KV is the best default** — no measurable accuracy loss, no throughput hit.
   - **TurboQuant `4bit_nc`** is the only TQ variant viable for production.
   - **TurboQuant `3bit_nc` / `k3v4_nc`** drop 15–25 pts on reasoning at long context.
   - TurboQuant `k8v4` is dominated by FP8.
   - Skipping the first and last two layers from compression preserves most of the
     quality at almost the same compression ratio.
2. **Community ports independently converged on dropping QJL.** [tonbistudio's V3
   PyTorch port](https://github.com/tonbistudio/turboquant-pytorch), the
   [scos-lab benchmark across 8 models](https://github.com/tonbistudio/turboquant-pytorch/issues/8),
   and [0xSero's Triton kernels](https://github.com/0xSero/turboquant) all find that
   the 1-bit QJL residual *adds* error at b ≤ 3 because softmax amplifies its variance.
   The `*_nc` ("norm correction") variants that landed in vLLM are the official answer.
3. **The serving stack consolidated.** vLLM merged
   `--kv-cache-dtype turboquant_{k8v4, 4bit_nc, k3v4_nc, 3bit_nc}` upstream. A production
   fork ([`turboquant-plus-vllm`](https://pypi.org/project/turboquant-plus-vllm/)) ships
   CUDA dequant kernels with **10.1× decode speedup on Qwen3.6-35B-A3B-TQ3**. llama.cpp
   has two forks ([spiritbuun](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)
   for head_dim=128, [AmesianX](https://github.com/AmesianX/TurboQuant) for any head_dim).
   SGLang remains WIP with three competing PRs.

---

## TL;DR — One-line positioning

| Method | Axis | Best at | TurboQuant relationship |
|---|---|---|---|
| **FP8 (E4M3) KV** | Hardware FP8 container | **Best default in May 2026** — 2× compression, no measurable loss | **Strictly better than TQ when 2× is enough** |
| **TurboQuant `4bit_nc`** (this repo) | **Rotation + scalar VQ, no QJL** | **Data-oblivious 2.6–3.1× compression** when FP8 doesn't fit | The recommended TQ variant |
| **TurboQuant `3bit_nc`** | Rotation + scalar VQ, no QJL, 3 bpv | Memory-bound edge / consumer GPUs | Accept 15–25 pt drops on reasoning at long context |
| **TurboQuant (paper, w/ QJL)** | Rotation + scalar VQ + 1-bit QJL residual | The published algorithm | Worse than `*_nc` in independent reproductions |
| **NVFP4 / MXFP4 KV** | Hardware-native FP4 block quant | Blackwell/SM100/SM120 throughput on large serving rigs | **Complementary** — TQ is the algorithm, NVFP4 is the container |
| **TriAttention** (Apr 2026) | Pre-RoPE Q/K concentration → trigonometric importance scoring | **Long reasoning (AIME-style 32K CoT)** via eviction, up to 10.7× KV memory reduction | **Orthogonal** — TQ compresses every retained token, TriAttention decides which ones |
| **Adaptive KV-quant** (Apr 6, 2026) | Learned per-token bit-width {2, 4, 8, FP16} | On-device / edge LLMs where static bits waste budget | Can wrap TQ as the 2-bit/4-bit backend |
| **LRKV (Low-Rank KV)** (Apr 9, 2026) | Shared full-rank basis + head-specific low-rank residual | Training-time architectural fix (45–53% KV reduction) | **Architectural** — applies at pretraining; TQ applies at inference |
| **DeepSeek MLA** | Multi-head Latent Attention: down-projection of K/V | Architectural compression, native to the model weights | Model-side; TQ still compresses the residual latent KV |
| **KIVI** (ICLR'24) | Per-channel K + per-token V asymmetric 2-bit | Simple, calibration-free 2-bit baseline | Predecessor; TQ `4bit_nc` is better on long-context |
| **KVQuant** | Per-channel + pre-RoPE K + dense-sparse split | Sub-4-bit with outlier preservation | Outlier handling similar in spirit to TQ's two-rotation split |
| **ParoQuant** (ICLR'26, Apr 2026) | Scaled pairwise rotation for **weight** quant | INT4 **weights** with +2.4% over AWQ on reasoning | **Weight-side, not KV-side** — stack with TurboQuant |
| **SnapKV / H2O / PyramidKV / StreamingLLM** | Importance-based token eviction | High-ratio drop for long context (>100K) | **Orthogonal** — combine with TQ for multiplicative savings |
| **KV Packet** (Apr 14, 2026) | Context-independent cache reuse via soft-token adapters | Cross-document KV reuse in RAG without recompute | Orthogonal — caches TQ-compressed packets across sessions |
| **NVIDIA KVPress** | Framework of press strategies (ExpectedAttention, ThinK, etc.) | Plug-in eviction & budget control | Orthogonal — KVPress can drive which tokens TQ keeps |
| **[kvtc](https://github.com/OnlyTerp/kvtc)** (sibling project) | PCA-decorrelated channel rotation + per-channel scalar quant | Calibration-based KV compression when a few samples of real activations are available | **Sibling** — same authors, data-**aware** variant; TurboQuant stays data-*oblivious* for zero-setup deployments |

The short summary: **TurboQuant is a per-token precision compressor; the other axis is
token selection/eviction; the third axis is hardware format (FP4/FP8/INT4).** The three
compose — but in May 2026, FP8 is the precision-axis default unless you specifically need
more than 2× compression.

---

## What's new (May 2026)

### May 11–18, 2026

- **May 18 — [Towards AI long-context follow-up](https://pub.towardsai.net/)** independent
  verification on Llama-3.1-70B that 3-bit TQ degrades dramatically beyond 32K context.
- **May 17 — [AI Intensify: "TurboQuant — is compression and performance worth the hype?"](https://ai-intensify.com/turboquant-is-compression-and-performance-worth-the-hype/)**
  Evaluation against FP8 across three models. Same conclusion as Red Hat: FP8 is the
  better default; TQ wins only when 2× isn't enough.
- **May 16 — [scos-lab/turboquant 8-model benchmark](https://github.com/tonbistudio/turboquant-pytorch/issues/8)**
  Pure-MSE beats MSE+QJL on every model tested. Establishes the K/V-norm-ratio metric as
  the single best predictor of TQ compression quality.
- **May 14 — [TeqVolt: "TurboQuant — Google's 6× KV Cache Compression"](https://teqvolt.com/deep-dives/turboquant-google-6x-kv-cache-compression-3-bit-zero-loss)**
  Third-party deep-dive on the algorithm. Notes the paper's "zero loss" claim doesn't
  generalize cleanly to frontier MoE models.
- **May 13 — [varjoranta/turboquant-vllm v0.13.5](https://github.com/varjoranta/turboquant-vllm/releases/tag/v0.13.5)**
  End-to-end Qwen3.6-35B-A3B-TQ3-native support through vLLM 0.20.2 + FlashInfer CUTLASS
  MoE. First native-packed TQ checkpoint for a frontier model.
- **May 11 — [**Red Hat AI / vLLM blog: "A First Comprehensive Study of TurboQuant"**](https://blog.vllm.ai/2026/05/11/turboquant.html)**
  by Eldar Kurtić, Michael Goin, Alexandre Marques. The most important data point in the
  current ecosystem. Frames TurboQuant against the FP8 KV baseline rigorously for the
  first time.

### Late April 2026

- **Apr 23 — [varjoranta/turboquant-vllm v0.13.0](https://github.com/varjoranta/turboquant-vllm/releases/tag/v0.13.0)**
  Block-diagonal WHT CUDA kernel: 10.1× decode speedup on Qwen3.6-35B-A3B-TQ3 versus the
  reference PyTorch path.
- **Apr 22 — [vLLM merges canonical TurboQuant KV dtypes](https://docs.vllm.ai/en/v0.20.2/api/vllm/model_executor/layers/quantization/turboquant/)**
  `--kv-cache-dtype turboquant_{k8v4, 4bit_nc, k3v4_nc, 3bit_nc}` ships in vLLM 0.20.x.
  These names supersede the earlier `tq3` / `tq4` / `pq4` proposals.
- **Apr 18 — [AmesianX/TurboQuant v1.3.0](https://github.com/AmesianX/TurboQuant/releases/tag/v1.3.0)**
  P1→P5 head_dim detection cascade lands; fixes head_dim=128 / 256 / 384 confusion on
  Qwen3.5, Qwen3.6, Gemma 4. Critical finding: `n_embd/n_head` lies for those models.
- **Apr 25 — [AmesianX v1.4.2](https://github.com/AmesianX/TurboQuant/releases/tag/v1.4.2)**
  MMA tensor-core acceleration on the inverse-WHT dequant path: 30 → 49 t/s (+63%) on
  GeForce RTX 5090 with Qwen3.6-35B-A3B.

### Apr 14–17, 2026

- **Apr 15 — [LMCache blog: "What is TurboQuant and why it matters for LLM inference"](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/)**
  by Kuntai Du (Tensormesh). The best layperson explainer of the algorithm. Frames
  TurboQuant as an answer to Jensen Huang's GTC 2026 warnings about KV cache memory
  dominating inference cost. Worth reading even though the May 2026 Red Hat evaluation
  has since added important caveats.
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
  optionally a QJL 1-bit residual for unbiased inner product.
- **Strengths:** **data-oblivious**, no calibration, zero compounding across layers
  (scalar per-coordinate, no coupling), provably within 2.7× of information-theoretic
  optimal distortion.
- **Benchmarks (paper):** 3.5 bpv = identical LongBench on Llama-3.1-8B; 2.5 bpv =
  marginal degradation. **But:** the [Red Hat May 2026 evaluation](https://blog.vllm.ai/2026/05/11/turboquant.html)
  finds 3-bit configurations drop 15–25 pts on reasoning + long-context retrieval, and
  the `*_nc` (no QJL) variants are uniformly better than the paper's `*_q` variants.
- **Variants shipped in vLLM** (May 2026):
  - `turboquant_k8v4` — 8-bit K, 4-bit V. Dominated by FP8 KV; rarely the right choice.
  - `turboquant_4bit_nc` — 4 bpv MSE-only, no QJL. **The recommended TQ config.**
  - `turboquant_k3v4_nc` — 3-bit K, 4-bit V, no QJL. For Qwen-class K-dominant models.
  - `turboquant_3bit_nc` — 3 bpv MSE-only. Edge / memory-bound only.
- **Weaknesses:** pure PyTorch reference is ~25× slower than FP16 on CPU; Triton kernels
  experimental; outlier handling differs slightly from paper (see
  [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)). For production, use
  [`turboquant-plus-vllm`](https://pypi.org/project/turboquant-plus-vllm/) or
  [0xSero/turboquant](https://github.com/0xSero/turboquant).

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
> just tells you where to start. Updated May 2026 to reflect the
> [Red Hat AI evaluation findings](https://blog.vllm.ai/2026/05/11/turboquant.html).

| Scenario | Start with | Then add | Why |
|---|---|---|---|
| **I just want something that works today** | **FP8 KV** (`--kv-cache-dtype fp8`) | — | Best default in May 2026 — no measurable accuracy loss, no throughput hit |
| **Datacenter Hopper/Blackwell, FP8 fits** | FP8 weights + **FP8 KV** | SnapKV / KVPress beyond 1M | Hardware-native, simplest, fastest |
| **Datacenter Hopper, FP8 doesn't fit** | FP8 weights + **`turboquant_4bit_nc`** | Skip first/last 2 layers | TQ earns its keep when you need >2× compression |
| **Datacenter Blackwell (B100/B200), max throughput** | NVFP4 weights + **FP8 KV** | — | Hardware-native FP4 path is the throughput king |
| **Single-card 70B+ at 128K+** | AWQ/NVFP4 + **`turboquant_4bit_nc`** | — | TQ 4bit_nc is roughly 1.5× more compression than FP8, just enough for these configurations |
| **Consumer Blackwell (RTX 5090, RTX PRO 6000 96GB)** | NVFP4 weights + **`turboquant_4bit_nc`** | — | RTX PRO 6000 96GB + TQ = Llama-4-Maverick-class on one card |
| **Consumer Ada (RTX 4090/4080) — no FP8 attention** | AWQ-INT4 weights + **`turboquant_4bit_nc`** | TriAttention / SnapKV for 32K+ reasoning | Ada lacks FP8 attention, so FP8 KV is no longer "free" |
| **Apple Silicon (M3/M4/M5)** | MLX-INT4 weights + TurboQuant via [varjoranta MLX port](https://github.com/varjoranta/turboquant-vllm) | — | MLX lacks NVFP4 + FP8 attention; TQ Metal kernels work |
| **On-device / edge (phone, Jetson)** | Adaptive KV-Quant or `turboquant_3bit_nc` | Token eviction (H2O / StreamingLLM) | Per-token bit allocation is essential at these budgets |
| **RAG systems, high cache reuse** | `turboquant_4bit_nc` KV | **KV Packet** or **LMCache** | Stack across-session reuse on top of per-token compression |
| **Reasoning / long CoT (AIME, AIME25, math)** | **FP8 KV** (3-bit TQ drops accuracy) | **TriAttention** | Red Hat eval shows 3-bit TQ degrades hard at long context; use FP8 + token selection instead |
| **Qwen3 / Qwen3.5 / Qwen3.6 (asymmetric K/V norms)** | **`turboquant_k3v4_nc`** | head_dim=256? use [AmesianX llama.cpp](https://github.com/AmesianX/TurboQuant) | Asymmetric K-heavy bit allocation matters for Qwen-family |
| **Pretraining a new model** | **LRKV** attention or **MLA** | TurboQuant at inference time | Architectural fix + inference compression multiply |

---

## Where TurboQuant fits in each serving engine

See [INTEGRATIONS.md](INTEGRATIONS.md) for concrete commands and links. Summary:

| Engine | Status (May 25, 2026) | How to use TurboQuant |
|---|---|---|
| **vLLM (upstream)** | **Merged** — see [docs.vllm.ai](https://docs.vllm.ai/en/v0.20.2/api/vllm/model_executor/layers/quantization/turboquant/) | `--kv-cache-dtype turboquant_{k8v4, 4bit_nc, k3v4_nc, 3bit_nc}` |
| **vLLM (production fork)** | [`turboquant-plus-vllm` v0.13.5](https://pypi.org/project/turboquant-plus-vllm/) — CUDA dequant kernels, MoE support, native checkpoints | `pip install turboquant-plus-vllm` |
| **vLLM (Triton reference)** | [0xSero/turboquant](https://github.com/0xSero/turboquant) | clone + `pip install -e .` |
| **SGLang** | WIP — competing PRs [#21419](https://github.com/sgl-project/sglang/pull/21419), [#22048](https://github.com/sgl-project/sglang/pull/22048), [#23135](https://github.com/sgl-project/sglang/pull/23135); none merged | `--kv-cache-dtype turboquant` on a PR branch |
| **llama.cpp (head_dim=128)** | [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) | `-ctk tq3 -ctv tq3 -fa` |
| **llama.cpp (any head_dim)** | [AmesianX/TurboQuant v1.4.2](https://github.com/AmesianX/TurboQuant) — supports head_dim=256+ (Qwen3.5, Qwen3.6, Gemma 4) | `-ctk tq3 -ctv tq3 -fa` |
| **PyTorch reference (QJL-free)** | [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) V3 | `pip install turboquant-pytorch` |
| **LMCache** | First-class; [explainer blog](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/) | Compose TQ-compressed tensors inside LMCache cross-session cache |
| **NVIDIA KVPress** | Eviction/selection framework | Use KVPress for token eviction, TQ for per-token precision |
| **MLX (Apple)** | [varjoranta Metal kernels](https://github.com/varjoranta/turboquant-vllm/tree/main/mlx) | `python -m turboquant.mlx.chat --model <id>` |

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

## Independent reproductions (May 2026)

The May 2026 evaluations are markedly less rosy than the paper. Headline numbers from
the [Red Hat AI / vLLM blog post (May 11)](https://blog.vllm.ai/2026/05/11/turboquant.html):

### Llama-3.3-70B-Instruct, MRCR long-context retrieval

| KV format | KV reduction vs BF16 | MRCR-8 needle (avg) | MRCR-16 needle (avg) | Comment |
|---|---|---|---|---|
| BF16 | 1.0× | 1.000 | 1.000 | baseline |
| **FP8 (E4M3)** | **2.0×** | **0.997** | **0.991** | **no measurable loss** |
| `turboquant_k8v4` | 2.0× | 0.989 | 0.973 | dominated by FP8 |
| `turboquant_4bit_nc` | 2.6× | 0.984 | 0.957 | the practical TQ choice |
| `turboquant_k3v4_nc` | 3.4× | 0.940 | 0.834 | usable, with care |
| `turboquant_3bit_nc` | 3.6× | 0.901 | 0.722 | large degradation |

### Qwen3-30B-A3B, AIME25 reasoning (Pass@1 over 32 samples)

| KV format | KV reduction | AIME25 acc | Comment |
|---|---|---|---|
| BF16 | 1.0× | **0.683** | baseline |
| **FP8** | **2.0×** | **0.680** | matches BF16 |
| `turboquant_k8v4` | 2.0× | 0.668 | small drop |
| `turboquant_4bit_nc` | 2.6× | 0.652 | tolerable |
| `turboquant_k3v4_nc` | 3.4× | 0.532 | ~15 pt drop |
| `turboquant_3bit_nc` | 3.5× | 0.443 | ~24 pt drop |

### Throughput on Qwen3-30B-A3B, single H100

| KV format | Tokens/s | vs BF16 |
|---|---|---|
| BF16 | 4,520 | 1.00× |
| **FP8** | **4,510** | **1.00×** (no penalty!) |
| `turboquant_4bit_nc` | 2,680 | 0.59× |
| `turboquant_3bit_nc` | 2,140 | 0.47× |

> **Why TurboQuant is slower than FP8 at decode:** FP8 quantizes both the KV cache *and*
> the attention compute on hardware FP8 tensor cores. TurboQuant only quantizes storage —
> the attention compute still happens in BF16/FP16 after dequantization. So you pay the
> dequant cost on every decode step without getting the compute speedup. The
> [varjoranta CUDA kernels](https://github.com/varjoranta/turboquant-vllm) and
> [0xSero Triton kernels](https://github.com/0xSero/turboquant) recover some of this gap
> through fused dequant + GEMV, but not all.

The community-port findings reinforce the same picture from a different angle:

- [tonbistudio V3 README](https://github.com/tonbistudio/turboquant-pytorch): "We
  initially shipped the paper's QJL-augmented MSE+sign quantizer. Our V3 ships pure MSE
  because, on every model we tested, the QJL step amplifies softmax noise and hurts
  end-to-end output quality."
- [scos-lab 8-model benchmark](https://github.com/tonbistudio/turboquant-pytorch/issues/8):
  "The K/V norm ratio is the single best predictor of compression quality. GPT-2-124M
  shows a 6× ratio; Qwen3.5-7B shows 106×; Qwen3.6-35B shows 182×. Models with extreme
  ratios benefit from asymmetric K-heavy bit allocation."
- [Towards AI (May 18)](https://pub.towardsai.net/): "Independent measurement across
  three implementations — sharpner/MLX, tonbistudio/PyTorch, 0xSero/Triton — found the
  same result: PolarQuant works, QJL doesn't help at b ≤ 3."

---

## Further reading

### Primary sources

- **Original paper:** [TurboQuant: Online Vector Quantization with Near-optimal Distortion
  Rate](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni; ICLR 2026).
- **ICLR 2026 poster:** [iclr.cc/virtual/2026/poster/10006985](https://iclr.cc/virtual/2026/poster/10006985)
  (archive — presented April 25, 2026).
- **Google Research blog:** ["TurboQuant: Redefining AI efficiency with extreme compression"](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
  (March 24, 2026).

### Companion papers

- [QJL (AAAI 2025)](https://dl.acm.org/doi/10.1609/aaai.v39i24.34773) — the 1-bit
  sign-based JL transform TurboQuant *can* use for its residual. Independent reproductions
  in May 2026 find this stage frequently hurts at b ≤ 3 (see above).
- [PolarQuant (arXiv 2502.02617)](https://arxiv.org/abs/2502.02617) — the recursive
  polar-coordinate predecessor TurboQuant deliberately avoids.

### Independent evaluations (May 2026)

- [Red Hat AI / vLLM: "A First Comprehensive Study of TurboQuant"](https://blog.vllm.ai/2026/05/11/turboquant.html) (May 11) — the definitive third-party eval.
- [TeqVolt: "TurboQuant — Google's 6× KV Cache Compression"](https://teqvolt.com/deep-dives/turboquant-google-6x-kv-cache-compression-3-bit-zero-loss) (May 14) — algorithm deep-dive.
- [AI Intensify: "Is compression and performance worth the hype?"](https://ai-intensify.com/turboquant-is-compression-and-performance-worth-the-hype/) (May 17) — three-model evaluation.
- [scos-lab/turboquant 8-model benchmark](https://github.com/tonbistudio/turboquant-pytorch/issues/8) — establishes the K/V-norm-ratio predictor.

### Production implementations

- [vLLM upstream `turboquant_*` dtypes](https://docs.vllm.ai/en/v0.20.2/api/vllm/model_executor/layers/quantization/turboquant/) — merged April 2026.
- [0xSero/turboquant](https://github.com/0xSero/turboquant) — Triton kernels, 1.1K stars.
- [varjoranta/turboquant-vllm](https://github.com/varjoranta/turboquant-vllm) — CUDA kernels, MoE support, MLX port.
- [AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant) — llama.cpp with head_dim=256+ support.
- [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) — V3 PyTorch port that drops QJL.
- [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) — llama.cpp CUDA, head_dim=128.

### Our own

- [README.md](README.md) — quickstart and headline numbers.
- [BENCHMARKS.md](BENCHMARKS.md) — memory tables, throughput targets, and methodology.
- [INTEGRATIONS.md](INTEGRATIONS.md) — vLLM / SGLang / llama.cpp / MLX / KVPress
  concrete commands.
- [FAQ.md](FAQ.md) — common misconceptions ("does this replace AWQ?", "FP8 vs TQ?", etc.).
- [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) — rotation modes, outlier
  channels, QJL projection, Blackwell stacking.
- [pseudocode.md](pseudocode.md) — line-by-line paper pseudocode for re-implementers.
