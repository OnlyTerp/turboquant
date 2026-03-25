"""
TurboQuant KV Cache Compression — End-to-End Demo

Demonstrates the complete TurboQuant pipeline:
  1. Creates a TurboQuantCache for a small model config
  2. Generates random Q, K, V tensors simulating a transformer
  3. Encodes K, V into the TQ cache
  4. Runs TQ attention and standard attention side by side
  5. Reports: MSE, cosine similarity, compression ratio, encode time, attention time

Works with pure PyTorch (no Triton required).

Usage:
    python demo.py
"""

import math
import os
import sys
import time

import torch
import torch.nn.functional as F

# Ensure the src directory is importable
sys.path.insert(0, os.path.dirname(__file__))

from cache import (
    TurboQuantCache,
    TurboQuantConfig,
    compression_ratio_fp16,
    memory_bytes_per_vector,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_LAYERS = 4
N_HEADS = 8
D = 128            # head dimension
SEQ_LEN = 64       # sequence length
N_QUERIES = 8      # number of queries to benchmark
DEVICE = torch.device("cpu")


def fmt_bar(char: str = "-", width: int = 72) -> str:
    return char * width

def fmt_row(label: str, value: str, width: int = 72) -> str:
    """Format a label-value row with right-aligned value."""
    return f"  {label:<{width - len(value) - 5}} {value}  "

# ---------------------------------------------------------------------------
# Main Demo
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 72)
    print(" TurboQuant KV Cache Compression - Demo ".center(72))
    print("=" * 72)
    print()

    # ── Step 1: Configuration ──────────────────────────────────────────
    print(" Configuration")
    print(fmt_bar("-"))
    print(f"  Layers:       {N_LAYERS}")
    print(f"  Heads:        {N_HEADS}")
    print(f"  Head dim:     {D}")
    print(f"  Sequence:     {SEQ_LEN}")
    print(f"  Device:       {DEVICE}")
    print()

    # ── Step 2: Compression Stats ──────────────────────────────────────
    ratio = compression_ratio_fp16(D)
    tq_bytes, fp16_bytes = memory_bytes_per_vector(D)
    total_fp16 = N_LAYERS * N_HEADS * SEQ_LEN * fp16_bytes * 2  # K + V
    total_tq = N_LAYERS * N_HEADS * SEQ_LEN * tq_bytes * 2

    print(" Compression Analysis")
    print(fmt_bar("-"))
    print(f"  FP16 per vector:        {fp16_bytes:>6} bytes")
    print(f"  TurboQuant per vector:  {tq_bytes:>6} bytes")
    print(f"  Compression ratio:      {ratio:>6.2f}×")
    print(f"  Bits per value:         {(tq_bytes * 8) / D:>6.2f}")
    print()
    print(f"  Total FP16 KV cache:    {total_fp16 / 1024:>8.1f} KB")
    print(f"  Total TQ KV cache:      {total_tq / 1024:>8.1f} KB")
    print(f"  Memory saved:           {(total_fp16 - total_tq) / 1024:>8.1f} KB ({100 * (1 - total_tq / total_fp16):.1f}%)")
    print()

    # ── Step 3: Create Cache ───────────────────────────────────────────
    print(" Creating TurboQuantCache...")
    t0 = time.perf_counter()
    cache = TurboQuantCache(N_LAYERS, N_HEADS, D, device=DEVICE)
    t_create = time.perf_counter() - t0
    print(f"   Created in {t_create * 1000:.1f} ms")
    print()

    # ── Step 4: Generate Random K, V ───────────────────────────────────
    torch.manual_seed(42)
    K_all = torch.randn(N_LAYERS, N_HEADS, SEQ_LEN, D, device=DEVICE)
    V_all = torch.randn(N_LAYERS, N_HEADS, SEQ_LEN, D, device=DEVICE)

    # ── Step 5: Encode into Cache ──────────────────────────────────────
    print(" Encoding K, V into TurboQuant cache...")
    encode_times = []
    t0 = time.perf_counter()
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            for t in range(SEQ_LEN):
                t_enc = time.perf_counter()
                cache.store(layer, head, K_all[layer, head, t], V_all[layer, head, t])
                encode_times.append(time.perf_counter() - t_enc)
    total_encode_time = time.perf_counter() - t0

    n_vectors = N_LAYERS * N_HEADS * SEQ_LEN
    avg_encode_us = (sum(encode_times) / len(encode_times)) * 1e6

    print(f"   Encoded {n_vectors:,} vectors (K + V)")
    print(f"   Total encode time:   {total_encode_time:.3f} s")
    print(f"   Avg per vector:      {avg_encode_us:.1f} us")
    print(f"   Throughput:          {n_vectors / total_encode_time:,.0f} vectors/sec")
    print()

    # ── Step 6: Benchmark Attention ────────────────────────────────────
    print(" Benchmarking attention...")
    print()

    # Pick layer 0, head 0 for detailed comparison
    layer, head = 0, 0
    K_ref = K_all[layer, head]  # [SEQ_LEN, D]
    V_ref = V_all[layer, head]  # [SEQ_LEN, D]

    cosine_sims = []
    mse_values = []
    tq_times = []
    std_times = []

    for q_idx in range(N_QUERIES):
        torch.manual_seed(1000 + q_idx)
        q = torch.randn(D, device=DEVICE)

        # Standard attention
        t0 = time.perf_counter()
        q_b = q.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, D]
        k_b = K_ref.unsqueeze(0).unsqueeze(0)             # [1, 1, S, D]
        v_b = V_ref.unsqueeze(0).unsqueeze(0)             # [1, 1, S, D]
        std_out = F.scaled_dot_product_attention(q_b, k_b, v_b).squeeze()
        std_time = time.perf_counter() - t0
        std_times.append(std_time)

        # TurboQuant attention
        t0 = time.perf_counter()
        tq_out = cache.compute_attention(layer, head, q, causal=False)
        tq_time = time.perf_counter() - t0
        tq_times.append(tq_time)

        # Metrics
        cos_sim = F.cosine_similarity(tq_out.unsqueeze(0), std_out.unsqueeze(0)).item()
        mse = F.mse_loss(tq_out, std_out).item()

        cosine_sims.append(cos_sim)
        mse_values.append(mse)

    # ── Step 7: Results Table ──────────────────────────────────────────
    avg_cos = sum(cosine_sims) / len(cosine_sims)
    min_cos = min(cosine_sims)
    avg_mse = sum(mse_values) / len(mse_values)
    avg_tq_us = (sum(tq_times) / len(tq_times)) * 1e6
    avg_std_us = (sum(std_times) / len(std_times)) * 1e6
    speedup = avg_std_us / avg_tq_us if avg_tq_us > 0 else float("inf")

    print("=" * 72)
    print(" Results Summary ".center(72))
    print("=" * 72)
    print(fmt_row("Compression ratio", f"{ratio:.2f}x"))
    print(fmt_row("Bits per value", f"{(tq_bytes * 8) / D:.2f}"))
    print(fmt_row("Memory saved", f"{100 * (1 - total_tq / total_fp16):.1f}%"))
    print("-" * 72)
    print(fmt_row("Vectors encoded", f"{n_vectors:,}"))
    print(fmt_row("Avg encode time", f"{avg_encode_us:.1f} us"))
    print(fmt_row("Total encode time", f"{total_encode_time:.3f} s"))
    print("-" * 72)
    print(fmt_row("Avg cosine similarity", f"{avg_cos:.6f}"))
    print(fmt_row("Min cosine similarity", f"{min_cos:.6f}"))
    print(fmt_row("Avg MSE (output)", f"{avg_mse:.6e}"))
    print("-" * 72)
    print(fmt_row("Avg TQ attention time", f"{avg_tq_us:.1f} us"))
    print(fmt_row("Avg FP16 attention time", f"{avg_std_us:.1f} us"))
    print("=" * 72)
    print()

    # ── Step 8: Per-query breakdown ────────────────────────────────────
    print("------------------------------------------------------------------------")
    print("  Query  |  Cosine Sim  |     MSE      |  TQ Time us  ")
    print("------------------------------------------------------------------------")
    for i in range(N_QUERIES):
        print(
            f"    {i:>2}   |  {cosine_sims[i]:>10.6f}  |  {mse_values[i]:>10.2e}  |  {tq_times[i]*1e6:>10.1f}  "
        )
    print("------------------------------------------------------------------------")
    print()

    # ── Step 9: Quality assessment ─────────────────────────────────────
    print(" Quality Assessment")
    print("-" * 72)
    if avg_cos >= 0.99:
        print("  Excellent - cosine similarity >= 0.99")
    elif avg_cos >= 0.95:
        print("  Good - cosine similarity >= 0.95")
    elif avg_cos >= 0.90:
        print("  Acceptable - cosine similarity >= 0.90")
    else:
        print("  Poor - cosine similarity < 0.90")

    if min_cos >= 0.95:
        print("  All queries above 0.95 threshold")
    elif min_cos >= 0.90:
        print("  Some queries below 0.95, but above 0.90")
    else:
        print("  Some queries below 0.90 - quality may be insufficient")

    print()
    print("Done.")
    print()


if __name__ == "__main__":
    main()
