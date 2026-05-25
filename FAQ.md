# TurboQuant FAQ

> Common questions and misconceptions, updated **2026-05-25**. For the broader 2026
> landscape, see [LANDSCAPE_2026.md](LANDSCAPE_2026.md). For concrete integrations, see
> [INTEGRATIONS.md](INTEGRATIONS.md).

---

## "Should I use FP8 or TurboQuant?"

**Default to FP8.** The [Red Hat AI / vLLM evaluation (May 11, 2026)](https://blog.vllm.ai/2026/05/11/turboquant.html)
is rigorous and conclusive on this question: across Llama-3.3-70B, Qwen3-30B-A3B, and
MiniMax-M2.7, on both long-context retrieval (MRCR) and reasoning (AIME25, GPQA, MATH500,
LiveCodeBench-v6), **FP8 (E4M3) KV cache** delivers:

- **No measurable accuracy loss** vs BF16
- **No throughput penalty** (4,510 t/s vs 4,520 t/s BF16 on Qwen3-30B-A3B / H100)
- **2× memory savings**

**Use TurboQuant only when:**

1. **2× isn't enough.** You need to fit a 70B+ model + cache on a single GPU at 128K+
   context, or you're memory-bound below the FP8 floor.
2. **Your hardware doesn't have FP8 attention.** On RTX 4090/4080 (Ada), older AMD,
   Apple Silicon, and most edge accelerators, FP8 KV doesn't give you the throughput
   bonus because the attention kernel still runs in BF16. TurboQuant's bandwidth savings
   are still useful there.

If you do reach for TurboQuant, use `turboquant_4bit_nc` — see the next question.

---

## "Which TurboQuant variant should I pick?"

```
vllm serve <model> --kv-cache-dtype turboquant_4bit_nc
```

**`turboquant_4bit_nc`** is the production-grade variant. It gives ~2.6–3.1×
compression (1.5× more than FP8) with quality within ~1 pt of FP8 on long-context evals.
The `_nc` suffix means "norm correction" — it drops the QJL residual in favor of a
simpler per-token norm rescale, which the community independently found to be uniformly
better at low bit widths.

The full ladder:

| Variant | KV reduction vs BF16 | Quality | When to use |
|---|---|---|---|
| `fp8` (not TQ, baseline) | 2.0× | Matches BF16 | **Always try first** |
| `turboquant_k8v4` | 2.0× | Slight drop vs FP8 | Rarely the right pick — FP8 wins |
| **`turboquant_4bit_nc`** | **2.6–3.1×** | Within 1–2 pt of FP8 | **The recommended TQ config** |
| `turboquant_k3v4_nc` | 3.4× | 10–15 pt drop on reasoning | Qwen-class with extreme K/V norm asymmetry |
| `turboquant_3bit_nc` | 3.5–4.0× | 15–25 pt drop on reasoning | Edge / strict memory budget only |

---

## "Does the QJL step actually help?"

**Usually not at b ≤ 3.** This is the most surprising finding from the May 2026
ecosystem. The paper's `TurboQuantProd` algorithm appends a 1-bit QJL (Quantized
Johnson-Lindenstrauss) sign residual to the MSE quantizer for an unbiased inner-product
estimator. In theory it's strictly better than MSE-only. In practice, every independent
reproduction has found the opposite at the bit widths people actually use:

- [tonbistudio V3 README](https://github.com/tonbistudio/turboquant-pytorch): "the
  paper's TurboQuantProd (QJL) for Keys gives +300% PPL at b=3 on GPT-2. MSE for both
  K and V gives only +7.6%."
- [scos-lab 8-model benchmark](https://github.com/tonbistudio/turboquant-pytorch/issues/8):
  pure MSE beats MSE+QJL on every model tested.
- [Red Hat AI](https://blog.vllm.ai/2026/05/11/turboquant.html): the `*_nc` (no-QJL)
  variants strictly Pareto-dominate the QJL-augmented variants.

**Why?** The QJL residual is *unbiased* but high-variance. Softmax in the attention
computation exponentially amplifies that variance, so the variance-bias trade-off doesn't
favor adding QJL at low bit widths. For Hadamard MSE quantizers at b ≥ 5 the paper's
analysis still holds — it's just that nobody is using b ≥ 5 because at that point you
might as well use FP8.

**What this repo defaults to:** the paper's algorithm (QJL on). Set `qjl_score_weight=0.0`
to switch to pure MSE — or use the `*_nc` variants in the vLLM `--kv-cache-dtype` flag.

---

## "Which layers should I skip from compression?"

The first and last 2 layers. The Red Hat evaluation found that these layers carry
disproportionate signal — quantizing them costs more accuracy than the bits they save.
vLLM does this automatically for the `*_nc` variants. If you're rolling your own
attention backend:

```python
SKIP = set(list(range(2)) + list(range(num_layers - 2, num_layers)))
# Keep FP16/FP8 KV for layers in SKIP; use TurboQuant for the rest.
```

---

## "How do I choose K vs V bits for my model?"

**Profile the per-channel K vs V norm ratio.** This single statistic predicts compression
quality better than any other metric the community has identified.

- Llama-class models: K/V norm ratio is roughly symmetric (1–3×). A uniform `4bit_nc`
  config works well.
- Qwen-class models: K/V norm ratio is **massively asymmetric** — Qwen3.5-7B sees a 106×
  ratio, Qwen3.6-35B-A3B sees 182×. K dominates the norm budget. Use `k8v4` or
  `k3v4_nc`.
- DeepSeek-V3 (MLA): the latent is small enough that K and V are roughly comparable —
  treat similarly to Llama-class.

To profile: run [`scos-lab/turboquant`](https://github.com/scos-lab/turboquant) on a few
batches of representative inputs and inspect the per-head K/V norm histograms. The K
budget should roughly match the K/V norm ratio in entropy bits.

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

**"3.5-bit" is a mode name from the paper, not a literal bit count.** In the default
"3.5-bit" configuration, TurboQuant splits channels into 32 outlier channels at 4
MSE bits and 96 regular channels at 3 MSE bits, plus 1 QJL residual bit per coordinate
— weighted-average bpv is `(32×4 + 96×3)/128 + 1 = 4.25` total, about 3.25 effective
after accounting for the small norm overhead (see [BENCHMARKS.md](BENCHMARKS.md#current-demo-results)).
The "3.5-bit" label is the paper's chosen name; the exact per-channel budget is controlled
by `MixedPrecisionConfig` in [`src/cache.py`](src/cache.py) via `b_mse` / `b_outlier` —
see [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for the exact outlier handling.
The default in `TurboQuantCache(mixed_precision=True, b_mse=2)` is actually closer to
the "2.5-bit" mode (3-bit outliers, 2-bit regular); bump `b_mse=3` to get the paper's
"3.5-bit" mode.

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

| Property | FP8 KV (E4M3) | KIVI (2-bit) | KVQuant | TurboQuant `4bit_nc` | TurboQuant `3bit_nc` |
|---|---|---|---|---|---|
| Bits/value | 8 | 2.25 | ~3.5 | 4.0 | 3.0 |
| Compression vs BF16 | **2.0×** | 7.1× | ~4.5× | 2.6–3.1× | 3.5–4.0× |
| MRCR-8 needle (Llama-3.3-70B) | **0.997** | not reported | not reported | 0.984 | 0.901 |
| AIME25 (Qwen3-30B-A3B) | **0.680** | — | — | 0.652 | 0.443 |
| Throughput penalty | **None** | High | Moderate | ~40% | ~50% |
| Requires calibration | No | No | Yes | **No** | **No** |
| Quantizes attention compute | **Yes (FP8 cores)** | No | No | No | No |
| Works on RoPE'd K | Yes | Per-channel helps | Pre-RoPE preferred | Yes | Yes |
| Best at | Default for most use cases | Older deployments | Sub-4-bit with calibration | Memory-bound long context | Edge / on-device |

Sources: [Red Hat AI eval (May 11, 2026)](https://blog.vllm.ai/2026/05/11/turboquant.html),
[TurboQuant paper](https://arxiv.org/abs/2504.19874).

**Why doesn't TurboQuant beat FP8 at the same quality?** FP8 quantizes both the KV
storage *and* the attention compute on hardware FP8 tensor cores (Hopper/Blackwell).
TurboQuant only quantizes storage — it dequantizes back to BF16 for the actual softmax
× value-projection compute. So FP8 wins on throughput *and* matches quality at 2×; TQ
only catches up in scenarios that demand more than 2× compression.

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

On Hopper/Blackwell hardware (May 2026, [Red Hat measurements](https://blog.vllm.ai/2026/05/11/turboquant.html)
on Qwen3-30B-A3B / H100):

| Metric | BF16 | **FP8 KV** | TQ `4bit_nc` | TQ `3bit_nc` |
|---|---|---|---|---|
| AIME25 | 0.683 | **0.680** | 0.652 | 0.443 |
| MRCR-8 (Llama-3.3-70B) | 1.000 | **0.997** | 0.984 | 0.901 |
| KV memory | 1.0× | **0.50×** | 0.32–0.38× | 0.25–0.29× |
| Tokens/s (decode) | 4,520 | **4,510** | 2,680 | 2,140 |
| Throughput penalty | 0% | **0%** | ~40% | ~50% |

**TurboQuant's throughput penalty** comes from a structural mismatch: FP8 quantizes both
KV *storage* and attention *compute* on hardware FP8 tensor cores. TurboQuant only
quantizes storage — the attention compute still runs in BF16 after dequantization. So
you pay the dequant cost on every decode step.

The paper-promised "memory bandwidth win" only materializes when you actually need
>2× compression (and therefore are memory-bandwidth bound, not compute bound). For most
serving setups, **FP8 wins on throughput, accuracy, and simplicity simultaneously**.

The CUDA-fused dequant kernels in [`turboquant-plus-vllm`](https://github.com/varjoranta/turboquant-vllm)
recover some of this gap (the v0.13.0 release reports 10.1× decode speedup on
Qwen3.6-35B-A3B over the reference path), but FP8 remains faster in the regime where
both fit.

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

## "Is TurboQuant the breakthrough of the year, like X/Twitter said in April?"

Short answer: **The April 2026 hype overshot. The May 2026 picture is more nuanced.**

In mid-April, the [LMCache blog post](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/)
paraphrased X/Twitter chatter calling TurboQuant "the most significant AI breakthrough
this year." That framing made sense given the paper's claims — data-oblivious, provably
near-optimal, 3.5 bpv at FP16 quality.

Then the [Red Hat AI evaluation in May](https://blog.vllm.ai/2026/05/11/turboquant.html)
ran the actual benchmarks and reported what independent ports had been finding for weeks:

1. **FP8 KV is a better default** than any TurboQuant config the team tested. It's free
   on Hopper/Blackwell.
2. **The 3-bit modes the paper headline-tests don't generalize cleanly** to frontier
   models. AIME25 drops 24 pts on Qwen3-30B-A3B with `turboquant_3bit_nc`.
3. **The QJL "residual" step the paper considers essential** actually hurts in practice.
   The `*_nc` variants drop QJL and outperform.

**Where TurboQuant is genuinely a contribution:**

- A clean, calibration-free quantization algorithm that *works* (especially `4bit_nc`).
- A useful tool for fitting models on memory-constrained hardware where 2× FP8 isn't
  enough.
- A mathematical framework (random rotation + scalar VQ) that's likely to influence
  future work even if the QJL step doesn't survive.

**Where the hype overshot:**

- It is not a universal replacement for existing KV quantization.
- The "3.5 bits at zero accuracy loss" claim doesn't hold up on frontier MoE models.
- It's a precision tool, not the precision tool — and FP8 wins the default slot.

The 2026 KV-compression landscape has several major shifts of comparable importance:
TriAttention (token selection, Apr 6), LRKV (architectural, Apr 9), Adaptive KV-Quant
(per-token bits, Apr 6), NVFP4 KV (hardware, Apr 2). The best 2026 stack combines them
along orthogonal axes — see [LANDSCAPE_2026.md §Decision guide](LANDSCAPE_2026.md#decision-guide-which-compressor-do-i-actually-want).

---

## "What's the right way to benchmark my model?"

The Red Hat evaluation uses the right methodology. Don't rely on cosine similarity
metrics on random vectors — they are useful sanity checks but **do not predict
end-to-end output quality**. The tonbistudio team published a retracted
"18/18 perfect generation at 5× compression" claim caused by a `residual_window=0` bug
that silently disabled compression; always log the *actual* compressed token count after
the run.

Concrete steps:

1. Pick your target engine — see [INTEGRATIONS.md](INTEGRATIONS.md).
2. Establish a baseline with **FP16/BF16 KV** *and* **FP8 KV** at your target context
   length. (You need both — FP8 is what TQ has to beat, not just FP16.)
3. Run two real evals at minimum:
   - **Long-context retrieval:** [MRCR](https://github.com/openai/simple-evals/blob/main/mrcr_eval.py) (multi-round needle) at 32K, 64K, 128K. The Red Hat blog uses MRCR-8 and MRCR-16.
   - **Reasoning:** AIME25 or LiveCodeBench-v6 Pass@1 over 32+ samples (these surface
     reasoning degradation that retrieval evals miss).
4. Switch to `turboquant_4bit_nc` (start here, not 3-bit) and re-run.
5. Expected delta: **< 1 pt** vs FP8 on Llama-3-class, **5–15 pt** on Qwen-class without
   asymmetric bit allocation. If you see worse, check:
   - Are first/last 2 layers being skipped from compression?
   - Are you using the `*_nc` (no-QJL) variant?
   - Is your model in the head_dim=128 family? If head_dim ≥ 256, use [AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant), not llama.cpp's spiritbuun fork.
   - Is `qjl_score_weight=0.0`? (Or for vLLM, are you on a `*_nc` dtype?)
   - Is your head dim a power of 2? (Hadamard rotation pads if not; pad tokens skew norms.)
6. If quality is still off and FP8 works fine — use FP8.

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

## "Where do I find the most up-to-date evaluations and explainers?"

Start here (sorted by depth):

**Definitive third-party evaluations:**

1. **[Red Hat AI / vLLM: "A First Comprehensive Study of TurboQuant" (May 11, 2026)](https://blog.vllm.ai/2026/05/11/turboquant.html)**
   — the rigorous evaluation. Read this before anything else.
2. **[scos-lab/turboquant 8-model benchmark (May 16, 2026)](https://github.com/tonbistudio/turboquant-pytorch/issues/8)**
   — establishes the K/V-norm-ratio predictor.

**Layperson / explainer:**

3. **[LMCache Blog (Apr 15, 2026)](https://blog.lmcache.ai/en/2026/04/15/what-is-turboquant-and-why-it-matters-for-llm-inference-in-laymens-term/)**
   — "in laymen's terms" explainer of the algorithm. The framing is more enthusiastic
   than the May 2026 data supports, but it's still the best layperson intro.
4. **[TeqVolt deep-dive (May 14, 2026)](https://teqvolt.com/deep-dives/turboquant-google-6x-kv-cache-compression-3-bit-zero-loss)**
   — algorithm walkthrough with up-to-date caveats.
5. **[AI Intensify hype check (May 17, 2026)](https://ai-intensify.com/turboquant-is-compression-and-performance-worth-the-hype/)**
   — three-model comparison against FP8.

**Hands-on:**

6. **[Towards AI: Running a 35B Model Locally with TurboQuant (Apr 15, 2026)](https://pub.towardsai.net/running-a-35b-model-locally-with-turboquant-whats-actually-possible-right-now-1ac5327430b0)**
   — consumer-GPU walkthrough.
7. **[MarkTechPost: NVIDIA KVPress end-to-end guide (Apr 9, 2026)](https://www.marktechpost.com/2026/04/09/an-end-to-end-coding-guide-to-nvidia-kvpress-for-long-context-llm-inference-kv-cache-compression-and-memory-efficient-generation/)**
   — how to stack KVPress eviction under a TurboQuant-style precision backend.

**Primary sources:**

8. **[The original paper (arXiv 2504.19874)](https://arxiv.org/abs/2504.19874)**
   — 14 pages, math-heavy.
9. **[Google Research blog (Mar 24, 2026)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)**
   — author-side framing.
10. **[ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10006985)** — presented Apr 25, 2026.
