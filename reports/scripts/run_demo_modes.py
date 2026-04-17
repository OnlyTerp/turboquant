"""Reproducible multi-mode TurboQuant demo.

Runs the CPU end-to-end demo at both the 2.5-bit and 3.5-bit mixed-precision modes,
records attention cosine similarity against FP16 `scaled_dot_product_attention` for
N_Q random queries, and writes a structured JSON report.

Usage:
    python reports/scripts/run_demo_modes.py

Output:
    reports/<YYYY-MM-DD>-demo-results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from cache import (  # noqa: E402
    N_OUTLIER_CHANNELS,
    TurboQuantCache,
    memory_bytes_per_vector,
)

N_LAYERS, N_HEADS, D, SEQ_LEN, N_Q = 4, 8, 128, 64, 16
DEVICE = torch.device("cpu")

MODES = [
    {"name": "2.5-bit mixed", "b_mse": 2, "b_outlier": 3},
    {"name": "3.5-bit mixed", "b_mse": 3, "b_outlier": 4},
]


def run_mode(mode: dict, K: torch.Tensor, V: torch.Tensor) -> dict:
    cache = TurboQuantCache(
        N_LAYERS,
        N_HEADS,
        D,
        b_mse=mode["b_mse"],
        device=DEVICE,
        mixed_precision=True,
        n_outlier=N_OUTLIER_CHANNELS,
        b_outlier=mode["b_outlier"],
    )
    tq_bytes, fp16_bytes = memory_bytes_per_vector(
        D,
        b_mse=mode["b_mse"],
        mixed_precision=True,
        n_outlier=N_OUTLIER_CHANNELS,
        b_outlier=mode["b_outlier"],
    )
    t0 = time.perf_counter()
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            for tok in range(SEQ_LEN):
                cache.store(layer, head, K[layer, head, tok], V[layer, head, tok])
    encode_time = time.perf_counter() - t0

    cos_sims, mses, std_times, tq_times = [], [], [], []
    layer, head = 0, 0
    K_ref, V_ref = K[layer, head], V[layer, head]
    for q_idx in range(N_Q):
        torch.manual_seed(1000 + q_idx)
        q = torch.randn(D, device=DEVICE)
        t0 = time.perf_counter()
        std_out = F.scaled_dot_product_attention(
            q.view(1, 1, 1, D), K_ref.view(1, 1, SEQ_LEN, D), V_ref.view(1, 1, SEQ_LEN, D)
        ).squeeze()
        std_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        tq_out = cache.compute_attention(layer, head, q)
        tq_times.append(time.perf_counter() - t0)
        cos_sims.append(
            F.cosine_similarity(tq_out.unsqueeze(0), std_out.unsqueeze(0)).item()
        )
        mses.append(F.mse_loss(tq_out, std_out).item())

    n_vec = N_LAYERS * N_HEADS * SEQ_LEN
    return {
        "mode": mode["name"],
        "b_mse": mode["b_mse"],
        "b_outlier": mode["b_outlier"],
        "bytes_per_vector": tq_bytes,
        "bpv": (tq_bytes * 8) / D,
        "ratio": fp16_bytes / tq_bytes,
        "encode_total_s": encode_time,
        "vectors_encoded": n_vec,
        "encode_us_per_vec": (encode_time * 1e6) / n_vec,
        "avg_cosine": sum(cos_sims) / len(cos_sims),
        "min_cosine": min(cos_sims),
        "max_cosine": max(cos_sims),
        "avg_mse": sum(mses) / len(mses),
        "avg_tq_us": sum(tq_times) * 1e6 / len(tq_times),
        "avg_std_us": sum(std_times) * 1e6 / len(std_times),
    }


def main() -> None:
    torch.manual_seed(42)
    K = torch.randn(N_LAYERS, N_HEADS, SEQ_LEN, D, device=DEVICE)
    V = torch.randn(N_LAYERS, N_HEADS, SEQ_LEN, D, device=DEVICE)

    results = []
    for mode in MODES:
        print(f"\n=== {mode['name']} (b_mse={mode['b_mse']}, b_outlier={mode['b_outlier']}) ===")
        result = run_mode(mode, K, V)
        print(json.dumps(result, indent=2))
        results.append(result)

    print("\n=== SUMMARY ===")
    for r in results:
        print(
            f"{r['mode']:16s}: bpv={r['bpv']:.2f}  ratio={r['ratio']:.2f}x  "
            f"cos_avg={r['avg_cosine']:.4f}  cos_min={r['min_cosine']:.4f}  mse={r['avg_mse']:.2e}"
        )

    out_path = os.path.join(REPO_ROOT, "reports", f"{date.today().isoformat()}-demo-results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "n_layers": N_LAYERS,
                    "n_heads": N_HEADS,
                    "d": D,
                    "seq_len": SEQ_LEN,
                    "n_queries": N_Q,
                    "device": "cpu",
                    "seed": 42,
                    "torch": torch.__version__,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
