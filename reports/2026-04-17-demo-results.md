# Fresh TurboQuant Demo Results — 2026-04-17

Reproducible pure-PyTorch (CPU) demo run at two mixed-precision modes. Random KV vectors,
synthetic attention, measures cosine similarity between TurboQuant attention output and
FP16 `scaled_dot_product_attention`. These replace the stale "0.90 cosine" numbers in
earlier drafts of BENCHMARKS.md — the QJL transpose bug was fixed upstream.

## Configuration

| | |
|---|---|
| n_layers | 4 |
| n_heads | 8 |
| head_dim (d) | 128 |
| seq_len | 64 |
| n_queries | 16 |
| device | CPU |
| torch | 2.11.0 + CUDA 13.0 build |
| seed | 42 |

## Results

| Mode | b_mse | b_outlier | bpv | Compression vs FP16 | Avg cosine | Min cosine | Max cosine |
|---|---|---|---|---|---|---|---|
| **2.5-bit mixed** | 2 | 3 | 3.62 | **4.41×** | **0.913** | 0.879 | 0.945 |
| **3.5-bit mixed** | 3 | 4 | 4.62 | **3.46×** | **0.975** | 0.955 | 0.986 |

Both modes use `mixed_precision=True` with 32 outlier channels (out of 128) given extra
bits, and `N_OUTLIER_CHANNELS=32` constant from `src/cache.py`.

- **bpv** includes the 1-bit QJL residual per coordinate **and** the per-vector norm
  overhead (16-bit float for the PQ norm + 16-bit float for the QJL residual norm). The
  raw bit budget at 2.5-bit is `(32×3 + 96×2)/128 + 1 = 3.25`; observed 3.625 bpv adds
  ~0.375 bpv for alignment/norm bookkeeping. At 3.5-bit: raw 4.25 → observed 4.625.

## How to reproduce

```bash
git clone https://github.com/OnlyTerp/turboquant.git
cd turboquant
pip install -e .
python src/demo.py                                    # 3.5-bit mode by default
python reports/scripts/run_demo_modes.py              # both modes, JSON artifact
```

Script + raw JSON: [`2026-04-17-demo-results.json`](2026-04-17-demo-results.json).

## Interpretation

- **3.5-bit mode is the headline mode.** 0.975 avg cosine similarity between TQ and FP16
  attention output on random vectors, 0.955 min. Paper reports zero LongBench accuracy
  loss at this setting on Llama-3.1-8B-Instruct.
- **2.5-bit mode is the aggressive mode.** 0.913 avg cosine, 7.1× theoretical compression
  (4.41× observed with overheads). Paper reports marginal LongBench degradation (49.44 vs
  50.06).
- These are **synthetic** attention — real models (see
  [README.md § Our Implementation Results](../README.md#-our-implementation-results-mistral-7b-instruct-v03))
  hit 0.963 logit cosine at 3.5-bit and 0.956 at 2.5-bit on Mistral-7B, because the QJL
  residual statistics match real KV distributions closely.

## Caveats

- CPU encode throughput is ~60 vectors/sec (this is the pure-Python Triton-fallback path,
  not the fused CUDA/Triton path). Real GPU encode is ~10,000× faster — see the
  [RTX 5090 build report](2026-03-31-build-report.md) for Blackwell numbers.
- Cosine similarity on random unit vectors is a proxy; perplexity / top-1 accuracy on a
  real LM is the gold standard and is tracked separately by `src/test_real_model.py`.
