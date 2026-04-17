# TurboQuant Launch Kit

This file collects ready-to-post content for the **ICLR 2026 poster week** launch window
(Zandieh et al. poster on **Sat Apr 25, 11:15 AM PDT**, session 10006985).

Everything below is drafted so you can paste directly into the target platform. Swap the
first-person pronouns for whatever fits your account.

---

## Contents

- [Quick-hit social posts](#quick-hit-social-posts)
- [X / Twitter thread (16 tweets)](#x--twitter-thread)
- [Show HN post](#show-hn-post)
- [/r/LocalLLaMA post](#rlocalllama-post)
- [LinkedIn post (longer form)](#linkedin-post)
- [Blog post outline](#blog-post-outline)
- [One-pager (for DMs / investors / recruiters)](#one-pager-for-dms--investors--recruiters)
- [ICLR poster handout](#iclr-poster-handout-half-page)

---

## Quick-hit social posts

### Hook variant A — the headline number
> A 32 GB RTX 5090 running Qwen3.5-27B at **1.5M tokens of context**.
> Same GPU running FP16: OOMs at ~232K.
>
> This is TurboQuant (ICLR 2026). 3.5-bit KV cache, near-zero accuracy loss.
>
> Open source: https://github.com/OnlyTerp/turboquant

### Hook variant B — the "faster than FP16" flex
> Something weird happened.
>
> I compressed the KV cache 4.9× and prefill got **faster**.
>
> Not 20% faster. 24%. On the same GPU. On the same model.
>
> Memory-bandwidth-bound attention + smaller KV = win. Here's why ↓

### Hook variant C — the landscape framing
> 2026 shipped a stack for long-context inference that would have sounded
> insane a year ago:
>
> • TurboQuant → 3.5-bit KV, no calibration
> • TriAttention → 10.7× token selection
> • NVFP4 → hardware-native FP4
> • LRKV → 45% architectural KV reduction
>
> The real trick is they **compose**. Guide:
> https://github.com/OnlyTerp/turboquant/blob/master/LANDSCAPE_2026.md

---

## X / Twitter thread

> Paste-ready. 16 tweets. Feel free to collapse 11–13 for character limits.

**1/16** Google's TurboQuant (ICLR 2026, Sat Apr 25 poster) compresses your LLM's KV cache to 3.5 bits per value with zero measurable quality loss.

First open-source impl: https://github.com/OnlyTerp/turboquant

Here's why this is a bigger deal than it sounds. ↓

**2/16** The KV cache is the memory of your LLM. Every token you feed it becomes ~megabytes of stored key/value tensors that every future token has to read.

At 128K context, a 70B model's KV cache is bigger than the model weights. At 1M context, it's **5×** bigger.

**3/16** Jensen spent most of GTC 2026 saying the quiet part loud: *KV cache memory is the #1 bottleneck for long-context inference, not compute.*

Every major 2026 GPU feature (NVFP4, FP8 KV dtypes, MLA support) is downstream of this realization.

**4/16** The field has been trying to compress the KV cache for two years:

• KIVI (2-bit, needs per-channel calibration)
• KVQuant (~4-bit, outlier handling)
• SnapKV / H2O (evict low-importance tokens)
• MLA (DeepSeek, architectural)

Each has tradeoffs. TurboQuant side-steps all of them.

**5/16** The TurboQuant idea is one of those "why didn't anyone do this first" things:

1. Random rotation of the KV vector → each coordinate independently ~N(0, 1/d)
2. Scalar Lloyd-Max quantize each coordinate
3. 1-bit QJL residual for the unbiased inner product

Zero calibration. Zero training. Works online.

**6/16** Provably within **2.7×** of the information-theoretic rate-distortion lower bound at 3.5 bits. That's not a benchmark — that's a theorem.

And it matches FP16 LongBench performance on Llama-3.1-8B-Instruct (50.06 vs 50.06).

**7/16** Numbers from our repo, verified on an RTX 5090 running Qwen3.5-27B:

• 32K prefill: **1.24× faster** than FP16 (same GPU, same model)
• 64K prefill: FP16 **OOMs**, turbo3 runs at 2,498 tok/s
• Max context on 32 GB: FP16 ~232K, **turbo2 = 1.5M**

**8/16** Yes, at long contexts TurboQuant is *faster* than FP16.

Attention in decode is memory-bandwidth bound. 4.9× less KV data → the HBM read is the bottleneck, not the decode. The decode-overhead "tax" is ~0.5 µs/query on H100.

**9/16** What you actually stack it with (the 2026 answer):

• Weights: NVFP4 (Blackwell) / AWQ-INT4 (Ada) / FP8 (Hopper)
• KV precision: **TurboQuant 3.5-bit**
• Token selection: TriAttention or SnapKV for 32K+ CoT
• Cache reuse: KV Packet or LMCache for RAG

All four axes compose.

**10/16** Engine support as of Apr 17, 2026:

• vLLM: PR #39890 adds `--kv-cache-dtype turboquant_3bit`
• SGLang: PR #21419 adds `--kv-cache-dtype turboquant`
• llama.cpp: b8779 DP4A flash-attn, native TQ path in progress
• KVPress: stack TQ under any "press" strategy
• LMCache: first-class

**11/16** The "why now" is that the paper dropped Apr 2025, got the ICLR 2026 acceptance, and over the past 10 weeks nearly every serving-engine team integrated it.

LMCache called it "possibly the most significant AI breakthrough this year." Towards AI did a full hands-on 35B walkthrough last week.

**12/16** Crucially: **TurboQuant does not quantize weights.**

It stacks **on top of** your weight quantization. AWQ/GGUF/NVFP4 users: you don't replace your weights quant, you add TurboQuant on the KV cache and compound the savings.

**13/16** The 2026 landscape also has architectural KV reductions:

• DeepSeek MLA (shipped, latent down-projection)
• LRKV (Apr 9, 2026): 45–53% via shared basis + per-head low-rank residual
• GQA (you already use this)

These are model-side. TurboQuant is inference-side. They compose.

**14/16** For anyone doing long CoT reasoning (o3-style, AIME-style, o1-clone training): TriAttention (arXiv 2604.04921, Apr 11, 2026) does 10.7× reduction by picking the trig-important tokens.

TriAttention decides *which* tokens survive. TurboQuant compresses *what survives*.

**15/16** What's still open:

• Fused Triton kernels competitive with FlashAttention-4 (in progress)
• Upstream llama.cpp PR for native turboquant K-type/V-type (drafting)
• HuggingFace-hosted precomputed outlier channel lists per model

PRs welcome.

**16/16** Full guide — ICLR poster, paper, all 15+ competing methods side-by-side, per-engine setup, FAQ:

https://github.com/OnlyTerp/turboquant

ICLR 2026 poster: Zandieh et al., Sat Apr 25, 11:15 AM PDT.

Zero-hype take: this is how long context actually gets affordable.

---

## Show HN post

**Title:** Show HN: TurboQuant – 3.5-bit KV cache compression for LLMs (ICLR 2026)

**Body:**

Hi HN,

TurboQuant (Zandieh et al., Google Research / NYU, ICLR 2026) compresses the KV cache of
a transformer LM down to 3.5 bits per value with no measurable quality loss on LongBench.
This repo is the first open-source implementation — reference PyTorch code, a vLLM
plugin, and a working hand-off into SGLang and llama.cpp forks.

Why it works (short version): after a random Hadamard rotation, each coordinate of a KV
vector follows a near-Gaussian independent of the others, so scalar Lloyd-Max
quantization is near-optimal per-coordinate. A 1-bit QJL sign-projection residual gives
the unbiased inner-product correction that makes attention scores accurate. Provably
within 2.7× of the rate-distortion lower bound.

What makes it interesting beyond "yet another quantizer":

- **Zero calibration.** Codebooks are fixed at init; outlier channels are detected
  online. You can drop it into an existing model without any profiling data.
- **At long context it's *faster* than FP16.** Attention decode is memory-bandwidth
  bound. 4.9× smaller KV → the HBM read is the bottleneck, not the decode. On an RTX
  5090 I measured 1.24× prefill speedup at 32K on Qwen3.5-27B, and FP16 OOMs at 64K
  where turbo3 still runs. Max context before OOM on 32 GB went from ~232K → 1.5M.
- **Stacks with everything.** It does not touch weights; it's orthogonal to AWQ/GGUF/
  NVFP4. It's orthogonal to token eviction (SnapKV, TriAttention). It's orthogonal to
  architectural KV reductions (MLA, LRKV). The paper and the repo give the composition
  rules.

What's in the repo:

- `src/cache.py` — the reference PyTorch TurboQuant cache with two rotation modes
  (Hadamard O(d log d) and dense QR)
- `vllm_plugin/` — drop-in attention backend that routes vLLM's KV cache through
  TurboQuant
- `LANDSCAPE_2026.md` — a full compare-and-contrast against TriAttention, LRKV,
  Adaptive KV-Quant, ParoQuant, NVFP4-KV, KVPress, KIVI, KVQuant, KV Packet, SnapKV,
  H2O, StreamingLLM, and DeepSeek MLA
- `INTEGRATIONS.md` — tested commands for vLLM, SGLang, llama.cpp, KVPress, LMCache,
  MLX, Transformers, Docker
- `BENCHMARKS.md` — reproducible CPU demo numbers + the RTX 5090 hardware report

Paper: https://arxiv.org/abs/2504.19874
Repo: https://github.com/OnlyTerp/turboquant
ICLR 2026 poster: Zandieh et al., Sat Apr 25, 11:15 AM PDT

Happy to answer questions about the math (Lloyd-Max, QJL variance bounds, rotation-mode
tradeoffs), engine integration, or how this stacks with what you're already running.

---

## /r/LocalLLaMA post

**Title:** RTX 5090 + Qwen3.5-27B at 1.5M context using 3.5-bit TurboQuant KV cache (ICLR 2026)

**Body:**

Spent the week testing TurboQuant (the new ICLR 2026 KV-cache compression method from
Google Research) on a single 32 GB RTX 5090. Results are honestly kind of absurd.

**Qwen3.5-27B, Q4_K_M weights, flash attention, same GPU:**

| Config | Prefill @ 32K | Max context before OOM | tg128 gen |
|---|---|---|---|
| f16 KV | 2,482 tok/s | ~232K | 70.22 tok/s |
| **turbo3 KV** (~3.5 bpv) | **3,068 tok/s (1.24×)** | **~1.1M** | 67.77 tok/s |
| **turbo2 KV** (~2.5 bpv) | similar | **~1.5M** | **71.10 tok/s** |

Yes you read that right — at 3.5-bit KV with FA on, prefill is 24% *faster* than FP16 on
the same hardware. At 64K+ context FP16 simply doesn't fit and TurboQuant keeps running.

And if you look at generation throughput, turbo2 (the most aggressive compression) is
actually slightly faster than FP16, because attention decode is memory-bandwidth bound
and you're reading less data.

**What I actually did:**

1. Cloned https://github.com/OnlyTerp/turboquant
2. Used the llama-cpp-turboquant fork on a Windows RTX 5090 (Blackwell SM120)
3. Configured `-ctk turbo3 -ctv turbo3 -fa` and raised `-c` gradually to find the OOM
   point
4. Numbers above are the median of three runs, llama-bench
5. Full log + rebuild steps in
   [`reports/2026-03-31-build-report.md`](https://github.com/OnlyTerp/turboquant/blob/master/reports/2026-03-31-build-report.md)

**Why this works (short version):**

Random Hadamard rotation → each coordinate becomes near-Gaussian + near-independent of
the others → scalar Lloyd-Max quantization is near-optimal → 1-bit QJL sign-projection
residual gives you the unbiased inner product for attention scores. Provably within
2.7× of the information-theoretic lower bound. No calibration, no training.

**What it stacks with:**

- ✅ AWQ / GGUF / NVFP4 weight quantization (TQ doesn't touch weights)
- ✅ SnapKV / TriAttention / H2O token eviction
- ✅ KVPress press strategies (ExpectedAttention, ThinK)
- ✅ FlashAttention, GQA, MLA, RoPE

**Status per engine:**

- vLLM: PR #39890 adds official `turboquant_3bit/_4bit` modes (merging soon)
- SGLang: PR #21419 adds `--kv-cache-dtype turboquant` with Triton kernels
- llama.cpp: DP4A flash-attn in b8779, native TQ type drafting
- KVPress: framework-level stack

Repo: https://github.com/OnlyTerp/turboquant
Paper: https://arxiv.org/abs/2504.19874
ICLR 2026 poster: Sat Apr 25, 11:15 AM PDT

Will answer questions below. Particularly interested to hear from anyone running MI300X
or Intel Gaudi — those backends are next on the integration list.

---

## LinkedIn post

> For a more professional audience (GTM, infra, ML platform engineers). ~350 words.

**Headline:** A 32 GB consumer GPU just ran 1.5M tokens of context. Here's how.

The long-context inference bottleneck in 2026 isn't compute — it's the KV cache. At
128K context a 70B model's KV cache is bigger than the model weights. At 1M context it's
five times bigger. Jensen Huang spent most of GTC 2026 saying this out loud; every
major GPU feature shipped this year (NVFP4, FP8 KV dtypes, MLA support) is downstream
of that realization.

TurboQuant (Zandieh et al., Google Research + NYU, ICLR 2026 poster Sat Apr 25)
compresses the KV cache to 3.5 bits per value with no measurable quality loss on
LongBench. The method is mathematically elegant: random Hadamard rotation makes each
coordinate near-Gaussian and near-independent, scalar Lloyd-Max quantization is then
provably near-optimal per-coordinate, and a 1-bit QJL sign-projection residual gives
unbiased inner products. Provably within 2.7× of the information-theoretic lower bound.

The practical consequences are striking. On a single 32 GB RTX 5090 running Qwen3.5-27B:

- FP16 KV cache: max ~232K context before OOM
- TurboQuant 3.5-bit KV: 1.1M context
- TurboQuant 2.5-bit KV: 1.5M context
- Prefill at 32K context: 1.24× **faster** than FP16 (same GPU, same model)

At long context, TurboQuant is faster than FP16, because attention decode is memory-
bandwidth bound and you're reading 4.9× less data.

Crucially, this composes. It does not touch model weights — you keep your AWQ/GGUF/
NVFP4 weight quantization. It composes with token eviction (TriAttention, SnapKV) and
with architectural KV compression (MLA, LRKV). The repo's `LANDSCAPE_2026.md` walks
through all 15+ methods side-by-side with per-scenario recipes.

We published the first open-source implementation at
https://github.com/OnlyTerp/turboquant with:

- Reference PyTorch cache
- vLLM plugin (upstream PR #39890 also landing this week)
- SGLang + llama.cpp integration notes
- Reproducible CPU demo + RTX 5090 hardware report

If you're running long-context inference on anything from RTX 4090 up to Blackwell
GB200, this is worth thirty minutes of your afternoon.

---

## Blog post outline

> Target: Substack / Medium / company blog. ~2000 words. Ship Apr 22–24 (ICLR week).

**Title options:**
- "TurboQuant, Explained: How LLMs Got a 5× Bigger Memory Without Getting Slower"
- "Why Long Context Is About to Get 10× Cheaper"
- "The 2026 KV Cache Compression Stack: A Field Guide"

**Structure:**

1. **Cold open (200 words).** RTX 5090 + Qwen3.5-27B running 1.5M tokens on 32 GB. One
   photo / one terminal gif. Setup: that would have been 5 GPUs 12 months ago.

2. **The bottleneck nobody noticed until late 2025 (300 words).** KV cache memory
   scaling arithmetic. Jensen's GTC 2026 quote. Why FP8 and NVFP4 on weights don't help
   the KV cache.

3. **The TurboQuant insight (400 words).** Random rotation → coordinate independence →
   scalar quantization is near-optimal. Explain Lloyd-Max as the MSE-minimizing quantizer.
   Why the 1-bit QJL residual preserves inner products for attention scores.
   Highlight: zero calibration, works online, provably within 2.7× of the information-
   theoretic bound.

4. **The numbers (300 words).** RTX 5090 benchmarks from the build report. Llama-3.1-8B
   paper LongBench numbers. The "faster than FP16 at long context" result and why
   memory-bandwidth-bound attention makes that possible.

5. **The 2026 landscape (500 words).** Three compression axes: precision × selection ×
   container. Walk through how TurboQuant, TriAttention, NVFP4, MLA, LRKV, KVPress, KIVI,
   and SnapKV each pick a different axis. Link to `LANDSCAPE_2026.md` for the full
   comparison.

6. **How to actually use it today (200 words).** vLLM / SGLang / llama.cpp one-liners.
   Link to `INTEGRATIONS.md`. Note ICLR poster.

7. **What's next (100 words).** Fused Triton kernels, native llama.cpp upstream,
   HuggingFace-hosted per-model outlier channel artifacts. Pull requests welcome.

**Image asks:**
- Diagram of the 3-stage pipeline (rotate → Lloyd-Max → QJL residual)
- Bar chart: KV memory at 128K for Llama-70B across {FP16, NVFP4, TurboQuant}
- Scatter: quality (LongBench) vs bits across {FP16, KIVI, KVQuant, TurboQuant, SnapKV}

---

## One-pager (for DMs / investors / recruiters)

**TurboQuant — open-source 3.5-bit KV cache compression for LLMs**

- Problem: At long context, KV cache memory is the dominant inference cost. A Llama-3-70B
  at 128K context uses 40 GB just for KV. Industry needs this to drop by 5×.
- Solution: TurboQuant (Zandieh et al., ICLR 2026). Rotation + scalar Lloyd-Max + 1-bit
  QJL residual. 3.5 bpv ≈ FP16 quality on LongBench. Provably within 2.7× of rate-
  distortion lower bound. No calibration.
- Evidence: RTX 5090 + Qwen3.5-27B: FP16 maxes at ~232K context, TurboQuant at 1.5M.
  Prefill 1.24× faster than FP16 at 32K.
- Status: First open-source impl live at github.com/OnlyTerp/turboquant. vLLM PR #39890
  and SGLang PR #21419 upstream. KVPress / LMCache / llama.cpp integrations. 33 unit
  tests, CI green.
- Asks: (a) contributor help on Triton kernel + llama.cpp upstream, (b) HF Hub hosting
  for precomputed per-model rotations, (c) industry partners running long-context
  workloads who'd like early bench access.

---

## ICLR poster handout (half-page)

> Print QR to the repo. Hand to anyone walking past the Zandieh poster who wants code.

**TurboQuant — 3.5-bit KV cache compression, open source**

- Paper: arXiv 2504.19874 (Zandieh, Daliri, Hadian, Mirrokni)
- Poster: this one, Sat Apr 25 11:15 AM PDT, session 10006985
- First open-source impl: github.com/OnlyTerp/turboquant
- Stack: works with vLLM, SGLang, llama.cpp, KVPress, LMCache
- Result: LongBench-parity with FP16 on Llama-3.1-8B. 1.24× prefill speedup at 32K on
  RTX 5090. 1.5M context on 32 GB.
- Repo has reference PyTorch code, vLLM plugin, full 2026 landscape comparison

[QR → github.com/OnlyTerp/turboquant]

---

## Coordination checklist

Before pushing any of the above:

- [ ] Confirm exact ICLR poster time / session ID (currently 10006985, Sat Apr 25 11:15 AM PDT)
- [ ] Coordinate timing with vLLM PR #39890 authors (don't claim "merged" if it isn't yet)
- [ ] Coordinate with SGLang PR #21419 authors
- [ ] Coordinate with paper authors on hand-off etiquette (they've been friendly)
- [ ] Check that the RTX 5090 numbers in the build report are still current after any
      kernel changes since 2026-03-31
- [ ] Have a blog host ready with the diagrams pre-rendered
- [ ] Pre-record a 60-second screen capture of the demo for X / LinkedIn reuse
