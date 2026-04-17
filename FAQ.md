# TurboQuant FAQ

> Common questions and misconceptions, updated **2026-04-17**. For the broader 2026
> landscape, see [LANDSCAPE_2026.md](LANDSCAPE_2026.md). For concrete integrations, see
> [INTEGRATIONS.md](INTEGRATIONS.md).

---

## "Is this a replacement for AWQ / GPTQ / GGUF?"

**No.** TurboQuant compresses the **KV cache** at inference time. It does not touch
model weights. It **stacks on top of** AWQ / GPTQ / GGUF / NVFP4 / MXFP4.

A typical 2026 production stack:

```
Weights:    INT4 (AWQ / GPTQ) or NVFP4 (Blackwell)   →  ~4× parameter memory savings
   ×
KV cache:   TurboQuant 3.5-bit                        →  ~4.9× KV memory savings
```

You multiply the savings. A Llama-3.1-8B serving rig that used to need 40 GB
(16 GB weights + 16 GB KV + overhead) at 128K context drops to roughly 8 GB.

See [Mustafa Genc's Towards AI walkthrough (Apr 15, 2026)](https://pub.towardsai.net/running-a-35b-model-locally-with-turboquant-whats-actually-possible-right-now-1ac5327430b0)
for a detailed breakdown of the Q4_K_M × TurboQuant stack.

---

## "Why 3.5 bits? Why not 2 bits or 8 bits?"

The paper shows three regimes:

| Bits | LongBench vs FP16 | Use when |
|---|---|---|
| **3.5** (default) | **Identical** (50.06 vs 50.06) | You want zero perceptible quality loss |
| **2.5** | Marginal degradation (49.44) | You need maximum memory savings and can tolerate <1 pt |
| 4.5 / 5.5 | Indistinguishable from 3.5 | Rarely worth it — diminishing returns |

The "3.5" is because TurboQuant splits channels into outliers (3 bits) and regular
(2 bits), so the weighted average is 3.5. `MixedPrecisionConfig` in [`src/cache.py`](src/cache.py)
controls this split — see [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for the
exact outlier handling.

---

## "Do I need to calibrate TurboQuant on my data?"

**No.** TurboQuant is **data-oblivious** — the random rotation and Lloyd-Max codebook
are fixed at initialization based only on the head dimension `d` and seed. This is the
key differentiator from KVQuant / SmoothQuant / AWQ-style methods, all of which require a
calibration dataset.

The only "online" piece is outlier-channel detection, which reads one batch of K/V
vectors to decide which 32 of the 128 channels get extra bits. This happens once per
layer/head at model load, not per-request.

---

## "How does this differ from KIVI / KVQuant / FP8 KV cache?"

| Property | FP8 KV | KIVI (2-bit) | KVQuant | **TurboQuant 3.5-bit** |
|---|---|---|---|---|
| Bits/value | 8 | 2.25 | ~3.5 | **3.5** |
| Compression vs FP16 | 2× | 7.1× | ~4.5× | **4.9×** |
| LongBench Avg (Llama-3.1-8B) | 50.0 | 48.50 | ~49.5 | **50.06** |
| Requires calibration | No | No | Yes | **No** |
| Unbiased attention scores | No | No | No | **Yes** (QJL residual) |
| Outlier handling | Per-tensor scale | Per-channel K | Dense-sparse split | **Two-rotation split** |
| Works on RoPE'd K | Yes | Per-channel helps here | Pre-RoPE preferred | Yes, post-RoPE |

The **unbiased inner product estimator** (the 1-bit QJL residual stage) is what makes
TurboQuant's attention scores mathematically unbiased, not just MSE-minimizing. This is
why it preserves needle-in-haystack retrieval at 128K context where the other methods
start to miss specific tokens.

---

## "Does TurboQuant work with RoPE?"

**Yes.** TurboQuant applies *after* RoPE has been applied to K (i.e., on the same K
tensor the attention kernel would see). The random rotation Π is orthogonal and applied
to the whole vector, so it doesn't interact with RoPE's per-pair rotation.

That said, for **eviction** methods (H2O, SnapKV, TriAttention) the story is different —
[TriAttention (Apr 2026)](https://arxiv.org/abs/2604.04921) specifically exploits the
**pre-RoPE** space where Q/K cluster around fixed centers. The methods are complementary:
TriAttention decides which tokens survive; TurboQuant compresses each survivor.

---

## "Does TurboQuant work with GQA / MQA / MLA?"

- **GQA / MQA** — yes, trivially. TurboQuant operates per-head independently, so sharing
  K/V across query heads is transparent.
- **MLA (DeepSeek-V3 style)** — yes. MLA stores a down-projected latent; TurboQuant
  compresses the latent vectors the same way it compresses standard K/V vectors.
- **LRKV ([Fin.AI, Apr 2026](https://fin.ai/research/low-rank-key-value-attention-reducing-kv-cache-memory-and-maintaining-head-diversity/))**
  — yes. LRKV stores a shared basis + per-head residual; both are vectors that can be
  TurboQuant-compressed.

---

## "Can I use TurboQuant with FlashAttention?"

- **FlashAttention-2** (H100 mainstream): The reference path doesn't fuse with FA2
  directly. Use the vLLM/SGLang plugin, which dequantizes the KV tile inside the attention
  kernel.
- **FlashAttention-3** (H100 optimized): Same story — use the engine plugin.
- **[FlashAttention-4](https://arxiviq.substack.com/p/flashattention-4-algorithm-and-kernel)**
  (Blackwell native, Mar 2026): FA4 was co-designed with FP4 tensor cores in mind. The
  natural stack on B200 is FA4 + **NVFP4 container** + **TurboQuant 3.5-bit encoding**.
  See [LANDSCAPE_2026.md §NVFP4 / MXFP4 KV](LANDSCAPE_2026.md#nvfp4--mxfp4-kv-blackwell-2026).

---

## "What hardware do I actually need?"

| Hardware | Status | Notes |
|---|---|---|
| **NVIDIA H100 / H200** (Hopper) | First-class | FP8 weights + TurboQuant KV is the standard datacenter combo in 2026 |
| **NVIDIA B100 / B200** (Blackwell, SM100) | First-class | Native FP4 → NVFP4 weights + TurboQuant KV |
| **NVIDIA RTX PRO 6000 Blackwell 96 GB** (SM120) | Working with workarounds | See [Allen Kuo (Apr 16)](https://allenkuo.medium.com/finishing-what-we-started-gemma-4-nvfp4-on-vllm-desktop-blackwell-wsl2-b2088c34815a) for the current WSL2/flashinfer workaround |
| **NVIDIA RTX 5090** (SM120) | Working with workarounds | Same SM120 path as RTX PRO 6000, smaller VRAM |
| **NVIDIA RTX 4090 / 4080** (Ada) | Fully supported | AWQ-INT4 + TurboQuant KV is the go-to consumer combo. [TriAttention enables single-4090 OpenClaw](https://arxiv.org/abs/2604.04921) |
| **NVIDIA A100** (Ampere) | Fully supported | FP8 not native; use INT8 weights + TurboQuant KV |
| **AMD MI300X / MI325X** | Supported via PyTorch | vLLM AMD backend works; ROCm Triton kernels a bit behind CUDA |
| **Apple M3 / M4 / M5** | PyTorch / MPS path | No FP4; [first M5 Max LLM benchmarks vs RTX PRO 6000](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/) |
| **Jetson Orin / Thor** | Edge path via [Adaptive KV-Quant](https://arxiv.org/abs/2604.04722) | Per-token bit allocation beats static at this budget |

---

## "Does TurboQuant introduce attention sinks / streaming issues?"

Not inherently. But most 2026 long-context deployments combine TurboQuant with an
**eviction** method like H2O, SnapKV, StreamingLLM, PyramidKV, or TriAttention. Those
methods already handle attention sinks (usually by pinning the first N tokens in FP16).

Attention-sink preservation for a pure-TurboQuant deployment is
[listed as a planned benchmark in BENCHMARKS.md](BENCHMARKS.md#attention-sinks-not-yet-implemented).
For now, if you're running beyond 64K context, stack TurboQuant with
[KVPress](https://github.com/NVIDIA/kvpress) or TriAttention.

---

## "What's the accuracy vs speed trade-off?"

On current hardware (Apr 2026):

| Metric | FP16 baseline | TurboQuant 3.5-bit | TurboQuant 2.5-bit |
|---|---|---|---|
| LongBench Avg (Llama-3.1-8B) | 50.06 | **50.06** | 49.44 |
| Needle @ 128K | 0.997 | **0.997** | 0.997 |
| KV memory | 1.0× | **0.20×** | 0.14× |
| H100 attention throughput (estimated, [BENCHMARKS.md](BENCHMARKS.md#attention-throughput-tokensssec)) | 1.0× | **4.4×** | 4.6× |
| RTX 5090 attention throughput | 1.0× | **4.3×** | 4.5× |
| On-paper distortion gap vs info-theoretic optimal | — | **2.7×** | 2.7× |

The throughput win comes from **memory bandwidth**: the attention kernel reads 52 bytes
per KV vector instead of 256.

---

## "Is TurboQuant overkill for short contexts (<8K)?"

Probably, yes. KV cache is a small fraction of memory at short contexts, so the
absolute GB saved is small. Stick with FP16 or FP8 KV for chat-bot workloads that stay
under 8K.

The crossover point where TurboQuant starts earning its keep:

- **Long context** (32K+): worth it.
- **Reasoning / chain-of-thought** (tens of thousands of generated tokens): worth it.
- **RAG with long documents**: worth it.
- **Chat under 4K**: marginal.

---

## "How does TurboQuant relate to the viral 'TurboQuant is the breakthrough of the year' takes on X?"

The quote is from the
[LMCache blog post (Apr 15, 2026)](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/),
paraphrasing discussion on X/Twitter. Our read:

- **The hype is largely earned.** Data-oblivious, provably near-optimal, no calibration,
  3.5 bpv at FP16 quality is genuinely new.
- **It's not the only 2026 breakthrough.** TriAttention (Apr 6), LRKV (Apr 9), Adaptive
  KV-Quant (Apr 6), and NVFP4 KV (Apr 2) are all major shifts. TurboQuant is the
  **precision** story; the others are **selection**, **architecture**, and **hardware**
  stories.
- **The best 2026 stack combines them.** See
  [LANDSCAPE_2026.md §Decision guide](LANDSCAPE_2026.md#decision-guide-which-compressor-do-i-actually-want).

---

## "What's the right way to benchmark my model?"

Use the paper's methodology — **LongBench + Needle-In-A-Haystack** at your target
context length. Cosine similarity on random vectors (what `src/demo.py` reports) is a
useful sanity check but not a substitute for end-to-end eval.

Concrete steps:

1. Pick your target engine (vLLM, SGLang, llama.cpp, KVPress) — see
   [INTEGRATIONS.md](INTEGRATIONS.md).
2. Run [LongBench](https://github.com/THUDM/LongBench) with FP16 KV to establish a
   baseline.
3. Switch to `turboquant_3bit` (or our plugin's default config) and re-run.
4. Expected delta on Llama-3.1-8B: **< 0.1 pts** on LongBench Avg, zero change on needle.
5. If you see larger deltas, check:
   - Is your outlier-channel detection actually being applied? (It's only applied when
     `MixedPrecisionConfig.enable=True`.)
   - Is `qjl_score_weight=1.0`? (Lower values trade bias for variance.)
   - Is your head dim a power of 2? (Hadamard rotation pads if not; pad token can skew
     norms.)

---

## "How do I cite this?"

Cite the **paper**, not this implementation:

```bibtex
@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

If you want to credit this open-source port specifically, you can link to the repo.

---

## "Where do I find the viral takes / explainers?"

Start here (sorted by how layperson-friendly they are):

1. **[LMCache Blog (Apr 15, 2026)](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/)**
   — "in laymen's term" explainer, no math. Best starting point.
2. **[Towards AI: Running a 35B Model Locally with TurboQuant (Apr 15, 2026)](https://pub.towardsai.net/running-a-35b-model-locally-with-turboquant-whats-actually-possible-right-now-1ac5327430b0)**
   — hands-on consumer-GPU walkthrough.
3. **[MarkTechPost: NVIDIA KVPress end-to-end guide (Apr 9, 2026)](https://www.marktechpost.com/2026/04/09/an-end-to-end-coding-guide-to-nvidia-kvpress-for-long-context-llm-inference-kv-cache-compression-and-memory-efficient-generation/)**
   — how to stack KVPress eviction under a TurboQuant-style precision backend.
4. **[ArXivIQ FlashAttention-4 analysis (Mar 13, 2026)](https://arxiviq.substack.com/p/flashattention-4-algorithm-and-kernel)**
   — FA4 paper summary relevant to Blackwell quantized KV.
5. **[The original paper (arXiv 2504.19874)](https://arxiv.org/abs/2504.19874)**
   — 14 pages, math-heavy but worth it.
6. **[ICLR 2026 poster (Apr 25, 2026 PDT)](https://iclr.cc/virtual/2026/poster/10006985)**
   — meet the authors.
