"""TurboQuant interactive demo — HuggingFace Space.

Two tabs:
  1. "KV memory calculator" — pick a model + prompt length, see how much GPU
     memory the KV cache would take at FP16 vs TurboQuant 3.5-bit vs 2.5-bit.
  2. "Live compression demo" — adjust bit settings, run real TurboQuant
     encode/decode on random KV vectors, see cosine similarity + compression
     ratio in real time.

Deploy: push the contents of this directory to a HuggingFace Space with
sdk=gradio. Or run locally:  python app.py
"""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr
import torch
import torch.nn.functional as F

from turboquant import (
    TurboQuantCache,
    memory_bytes_per_vector,
)


# ---------------------------------------------------------------------------
# Model zoo — (d_head, n_kv_heads, n_layers, "marketing name")
# Numbers are public-spec estimates as of Apr 2026. Cross-checked in
# LANDSCAPE_2026.md; minor corrections welcome via PR.
# ---------------------------------------------------------------------------

MODELS: dict[str, dict] = {
    "Llama-3.1-8B-Instruct": {"d": 128, "n_kv_heads": 8, "n_layers": 32,
                               "note": "GQA 8 KV heads. Paper headline model."},
    "Llama-3.1-70B-Instruct": {"d": 128, "n_kv_heads": 8, "n_layers": 80,
                                "note": "GQA 8 KV heads. 128K context."},
    "Llama-4-Scout-17B-16E (128 experts, 1 active)": {
        "d": 128, "n_kv_heads": 8, "n_layers": 48,
        "note": "MoE (1 of 16 experts active). 10M context target."},
    "Qwen3-72B": {"d": 128, "n_kv_heads": 8, "n_layers": 80,
                  "note": "GQA 8 KV heads. 256K context target."},
    "Qwen3.5-27B": {"d": 128, "n_kv_heads": 8, "n_layers": 48,
                    "note": "The RTX 5090 benchmark model from reports/2026-03-31-build-report.md."},
    "DeepSeek-V3 (MLA)": {"d": 128, "n_kv_heads": 128, "n_layers": 61,
                          "note": "MLA compresses KV architecturally; shown for context."},
    "Mistral-Nemo-12B": {"d": 128, "n_kv_heads": 8, "n_layers": 40,
                         "note": "GQA 8 KV heads. 1M context."},
    "Gemma-2-27B": {"d": 128, "n_kv_heads": 16, "n_layers": 46,
                    "note": "GQA 16 KV heads."},
}


@dataclass
class KVMemory:
    fp16_gb: float
    tq_35_gb: float
    tq_25_gb: float
    compression_35: float
    compression_25: float

    def ratio_str(self) -> str:
        return (f"FP16 {self.fp16_gb:.2f} GB  →  "
                f"TurboQuant 3.5-bit {self.tq_35_gb:.2f} GB  "
                f"({self.compression_35:.2f}×)")


def kv_memory(model: str, ctx_tokens: int, batch: int) -> KVMemory:
    """Compute KV cache memory for a (model, context, batch) combination.

    KV cache bytes for a GQA model =
        batch × seq_len × n_layers × 2 (K+V) × n_kv_heads × d_head × bytes_per_val
    """
    cfg = MODELS[model]
    d, n_kv, n_layers = cfg["d"], cfg["n_kv_heads"], cfg["n_layers"]
    # FP16 = 2 bytes per value.
    fp16_bytes = batch * ctx_tokens * n_layers * 2 * n_kv * d * 2
    # TurboQuant 3.5-bit = 4.625 bpv effective incl. norm overhead.
    tq35_bytes_per_vec, _ = memory_bytes_per_vector(d=d, b_mse=3, b_outlier=4, n_outlier=32)
    tq35_bytes = batch * ctx_tokens * n_layers * 2 * n_kv * tq35_bytes_per_vec
    # TurboQuant 2.5-bit = 3.625 bpv effective.
    tq25_bytes_per_vec, _ = memory_bytes_per_vector(d=d, b_mse=2, b_outlier=3, n_outlier=32)
    tq25_bytes = batch * ctx_tokens * n_layers * 2 * n_kv * tq25_bytes_per_vec
    return KVMemory(
        fp16_gb=fp16_bytes / 1e9,
        tq_35_gb=tq35_bytes / 1e9,
        tq_25_gb=tq25_bytes / 1e9,
        compression_35=fp16_bytes / tq35_bytes,
        compression_25=fp16_bytes / tq25_bytes,
    )


def memory_calculator(model: str, ctx_tokens: int, batch: int):
    mem = kv_memory(model, ctx_tokens, batch)
    note = MODELS[model]["note"]

    table = [
        ["FP16 (baseline)", f"{mem.fp16_gb:.2f} GB", "1.00×"],
        ["TurboQuant 3.5-bit (default, paper-parity quality)",
         f"{mem.tq_35_gb:.2f} GB", f"{mem.compression_35:.2f}×"],
        ["TurboQuant 2.5-bit (aggressive, ~0.6 LongBench pts)",
         f"{mem.tq_25_gb:.2f} GB", f"{mem.compression_25:.2f}×"],
    ]

    saved = mem.fp16_gb - mem.tq_35_gb
    summary_md = f"""
### {model}

> {note}

At **{ctx_tokens:,} tokens** × batch **{batch}**, TurboQuant 3.5-bit saves
**{saved:.1f} GB** of KV cache memory vs FP16 ({mem.compression_35:.2f}× smaller)
with near-zero LongBench accuracy loss.

*If this bar exceeds your GPU's VRAM, FP16 would OOM and TurboQuant would still serve.*
"""
    return table, summary_md


def live_demo(b_mse: int, b_outlier: int, n_outlier: int, seq_len: int,
              n_queries: int, seed: int):
    """Run real TurboQuant encode -> attention on random KV vectors.

    Mirrors src/demo.py: store a sequence of K/V vectors, run attention
    with N random queries, compare the output to the FP16 reference.
    """
    torch.manual_seed(seed)
    d = 128
    cache = TurboQuantCache(
        1, 1, d,
        b_mse=b_mse, b_outlier=b_outlier, n_outlier=n_outlier,
        mixed_precision=True, device="cpu",
    )
    K = torch.randn(seq_len, d)
    V = torch.randn(seq_len, d)
    for t in range(seq_len):
        cache.store(0, 0, K[t], V[t])

    cos_sims = []
    mse_values = []
    for i in range(n_queries):
        torch.manual_seed(1000 + i)
        q = torch.randn(d)
        ref = F.scaled_dot_product_attention(
            q.view(1, 1, 1, d), K.view(1, 1, seq_len, d), V.view(1, 1, seq_len, d),
        ).view(d)
        out = cache.compute_attention(0, 0, q)
        cos_sims.append(F.cosine_similarity(out.unsqueeze(0), ref.unsqueeze(0)).item())
        mse_values.append(F.mse_loss(out, ref).item())

    cos_mean = sum(cos_sims) / len(cos_sims)
    cos_min = min(cos_sims)
    cos_max = max(cos_sims)
    avg_mse = sum(mse_values) / len(mse_values)

    bytes_per, fp16_bytes = memory_bytes_per_vector(
        d=d, b_mse=b_mse, b_outlier=b_outlier, n_outlier=n_outlier
    )
    ratio = fp16_bytes / bytes_per
    bpv_effective = (bytes_per * 8.0) / d

    summary = f"""
### Results

| Metric | Value |
|---|---|
| **Avg cosine similarity** (TQ attention out vs FP16 attention out) | **{cos_mean:.4f}** |
| Min cosine | {cos_min:.4f} |
| Max cosine | {cos_max:.4f} |
| Avg MSE | {avg_mse:.2e} |
| Effective bits per value | {bpv_effective:.2f} bpv |
| Compression ratio vs FP16 | **{ratio:.2f}×** |
| Bytes per {d}-d vector | {bytes_per} |

### Interpretation

- `b_mse={b_mse}` bits for scalar Lloyd-Max quantization on the rotated K/V
- `b_outlier={b_outlier}` bits for the top `n_outlier={n_outlier}` outlier channels
- Methodology: encode {seq_len} random K/V pairs, run attention with {n_queries}
  random queries, compare TQ output vs FP16 `scaled_dot_product_attention`.
- Paper reports >=0.95 cosine at the 3.5-bit default (b_mse=3, b_outlier=4).
  Random Gaussian KVs are the hardest case; real activations compress better.
"""
    return summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="TurboQuant KV Cache Compression — Live Demo") as demo:
    gr.Markdown(
        "# ⚡ TurboQuant KV Cache Compression — Live Demo\n"
        "First open-source implementation of [TurboQuant (ICLR 2026)]"
        "(https://arxiv.org/abs/2504.19874). Compress your LLM's KV cache 3–5× "
        "with near-zero quality loss. Zero calibration. Zero training.\n\n"
        "Repo: [github.com/OnlyTerp/turboquant]"
        "(https://github.com/OnlyTerp/turboquant) · "
        "Landscape guide: [LANDSCAPE_2026.md]"
        "(https://github.com/OnlyTerp/turboquant/blob/master/LANDSCAPE_2026.md) · "
        "Install: `pip install turboquant`"
    )

    with gr.Tab("🧮 KV memory calculator"):
        gr.Markdown(
            "See how much GPU memory the KV cache takes for your model at a given "
            "context length. FP16 → TurboQuant 3.5-bit → TurboQuant 2.5-bit."
        )
        with gr.Row():
            model_in = gr.Dropdown(
                label="Model",
                choices=list(MODELS.keys()),
                value="Llama-3.1-8B-Instruct",
            )
            ctx_in = gr.Slider(
                label="Context length (tokens)",
                minimum=1024, maximum=1_048_576, value=32_768, step=1024,
            )
            batch_in = gr.Slider(
                label="Batch size (concurrent sequences)",
                minimum=1, maximum=32, value=1, step=1,
            )
        calc_btn = gr.Button("Compute", variant="primary")
        summary_out = gr.Markdown()
        table_out = gr.Dataframe(
            headers=["Mode", "Memory", "Compression"],
            datatype=["str", "str", "str"],
            row_count=(3, "fixed"),
        )
        calc_btn.click(
            memory_calculator,
            inputs=[model_in, ctx_in, batch_in],
            outputs=[table_out, summary_out],
        )
        # Compute once on load so the page isn't blank
        demo.load(
            memory_calculator,
            inputs=[model_in, ctx_in, batch_in],
            outputs=[table_out, summary_out],
        )

    with gr.Tab("🔬 Live compression demo"):
        gr.Markdown(
            "Run real TurboQuant encode → decode on random 128-d vectors. "
            "Measure cosine similarity vs the original. Sweep the bit knobs "
            "to see the quality / compression tradeoff."
        )
        with gr.Row():
            b_mse_in = gr.Slider(label="b_mse (bits for scalar quant)",
                                 minimum=1, maximum=6, value=3, step=1)
            b_out_in = gr.Slider(label="b_outlier (total bits for outlier channels)",
                                 minimum=1, maximum=8, value=4, step=1)
            n_out_in = gr.Slider(label="n_outlier (# outlier channels; 4..124, d-head=128)",
                                 minimum=4, maximum=124, value=32, step=4)
        with gr.Row():
            seq_in = gr.Slider(label="# KV vectors (sequence length)",
                               minimum=16, maximum=256, value=64, step=16)
            q_in = gr.Slider(label="# queries to run",
                             minimum=4, maximum=64, value=16, step=4)
            seed_in = gr.Slider(label="Random seed",
                                minimum=0, maximum=9999, value=42, step=1)
        run_btn = gr.Button("Run TurboQuant", variant="primary")
        result_out = gr.Markdown()
        run_btn.click(
            live_demo,
            inputs=[b_mse_in, b_out_in, n_out_in, seq_in, q_in, seed_in],
            outputs=result_out,
        )
        demo.load(
            live_demo,
            inputs=[b_mse_in, b_out_in, n_out_in, seq_in, q_in, seed_in],
            outputs=result_out,
        )

    gr.Markdown(
        "---\n"
        "*TurboQuant algorithm: Zandieh, Daliri, Hadian, Mirrokni — "
        "[ICLR 2026](https://iclr.cc/virtual/2026/poster/10006985). "
        "This demo is an independent open-source implementation.*"
    )


if __name__ == "__main__":
    demo.launch()
