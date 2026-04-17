---
title: TurboQuant KV Cache Demo
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: true
license: mit
short_description: Live demo of TurboQuant (ICLR 2026) — 3.5-bit KV cache compression
tags:
  - llm
  - inference
  - kv-cache
  - quantization
  - long-context
  - iclr2026
---

# TurboQuant KV Cache Demo

Live, interactive demo of **TurboQuant** ([ICLR 2026, Zandieh et al.](https://arxiv.org/abs/2504.19874)),
the near-optimal KV cache compressor that gets ≈FP16 quality at **3.5 bits per value**.

- **Paper**: https://arxiv.org/abs/2504.19874
- **Repo**: https://github.com/OnlyTerp/turboquant
- **Install**: `pip install turboquant`
- **2026 landscape guide**: [LANDSCAPE_2026.md](https://github.com/OnlyTerp/turboquant/blob/master/LANDSCAPE_2026.md)

## What the Space lets you do

1. **KV memory calculator** — pick a model (Llama-3.1, Qwen3, DeepSeek-V3 MLA,
   Gemma-2, Mistral-Nemo, Llama-4 Scout), set a context length and batch size,
   and instantly see how much GPU memory the KV cache would cost at FP16 vs
   TurboQuant 3.5-bit vs TurboQuant 2.5-bit.

2. **Live compression demo** — real TurboQuant encode/decode on random 128-d
   vectors, with sliders for `b_mse`, `b_outlier`, and `n_outlier`. Watch the
   quality ↔ compression tradeoff in real time.

## Deploying this Space

The directory `spaces/turboquant-demo/` in the upstream repo is the HF Space.
To deploy:

```bash
huggingface-cli login
huggingface-cli repo create --type space turboquant-demo --space_sdk gradio
cd spaces/turboquant-demo
git init && git remote add origin https://huggingface.co/spaces/OnlyTerp/turboquant-demo
git add . && git commit -m "initial" && git push -u origin main
```

Or push directly from the repo with the existing history (see the parent
repo's main README and RELEASING.md).

## Running locally

```bash
pip install turboquant gradio
python app.py
# -> http://127.0.0.1:7860
```

## Why not a notebook?

The notebook at [`notebooks/demo.ipynb`](https://github.com/OnlyTerp/turboquant/blob/master/notebooks/demo.ipynb)
is a better fit for reading through the algorithm step by step. This Space is
a better fit for "I just want to see the number for my model."
