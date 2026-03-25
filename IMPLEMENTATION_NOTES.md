# Implementation Notes

## Algorithm Choice: Scalar Quantization vs Recursive Polar Transform

This implementation follows **TurboQuant (arxiv 2504.19874)**, not PolarQuant (arxiv 2502.02617). While both papers share authors, they use fundamentally different approaches:

**TurboQuant (what we implement):**
- Random rotation → scalar Lloyd-Max quantization per coordinate
- Each coordinate quantized independently — no coupling between dimensions
- Errors do NOT compound through deep models
- Provably within 2.7× of information-theoretic optimal (Theorem 1)

**PolarQuant (what we DON'T implement):**
- Random preconditioning → recursive polar transform → angle quantization
- Coordinates coupled through sin/cos reconstruction tree (7 levels for d=128)
- Errors compound multiplicatively through the reconstruction tree AND through transformer layers
- Works well for shallow models but degrades on 32+ layer models

## Random Rotation

We provide **two rotation modes**, both satisfying the paper's requirement that P^T P = I:

### `rotation_mode="hadamard"` (default)
Randomized Hadamard Transform: Π = (1/√d) · H · D_signs
1. Random sign flip: multiply each coordinate by ±1 (deterministic from seed)
2. Fast Walsh-Hadamard Transform (FWHT): O(d log d) complexity
3. Scale by 1/√d

This is the practical O(d log d) variant. Requires power-of-2 dimension (we pad if needed).

### `rotation_mode="dense"`
Full random orthogonal matrix via QR decomposition of a Gaussian random matrix:
1. Generate A with i.i.d. N(0,1) entries
2. QR factorize: A = QR
3. Fix sign ambiguity for uniform Haar measure: Q ← Q · diag(sign(diag(R)))
4. Use Q as P

This is the theoretically exact approach — a uniformly random orthogonal matrix from the Haar measure on O(d). O(d²) storage and O(d²) per vector, but no padding needed.

### Why both?
The paper describes a general random orthogonal P and notes the Hadamard approach as a practical implementation. Both produce equivalent statistical properties: after rotation, each coordinate follows a Beta distribution converging to N(0, 1/d) for d ≥ 64 (Lemma 1), and distinct coordinates become **near-independent** in high dimensions. The dense mode is mathematically equivalent but slower; the Hadamard mode is the standard choice for production.

## Lloyd-Max Codebook

The scalar codebook is computed by solving the continuous 1D k-means problem (Eq. 3) for the Beta distribution f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2).

For d ≥ 64, we use the Gaussian approximation N(0, 1/d) which is faster and equally accurate. Optimal centroids for common bit widths:
- b=1: {±√(2/(πd))}
- b=2: {±0.453/√d, ±1.51/√d}

The **same codebook** is used for all coordinates — no per-level or per-channel variation needed.

## QJL Residual Correction

The QJL (Quantized Johnson-Lindenstrauss) stage provides unbiased inner product estimation:
1. Compute residual: r = x - DeQuant_mse(Quant_mse(x))
2. Project: sign(S · r) where S is a random matrix
3. Store: 1-bit signs + FP16 residual norm
4. Reconstruct: x̂_qjl = √(π/2)/d · ‖r‖ · S^T · signs

We use Rademacher (±1) entries for S instead of Gaussian N(0,1) for efficiency. Both satisfy the JL property; Rademacher avoids storing floating-point matrix entries.

## Outlier Channel Handling

The paper's 2.5-bit and 3.5-bit modes (Table 1) allocate more bits to outlier channels:
- 2.5-bit: 32 outlier channels at 3 bits + 96 regular at 2 bits
- 3.5-bit: similar split with higher bit allocation

**Paper approach (Section 2.3):** Split channels into outlier/regular sets, apply two **independent** TurboQuant instances with separate rotations and codebooks to each subset.

**Our implementation:** We follow the paper's approach exactly:
1. Detect outlier channels by **original-space variance** (before rotation) — the top-k highest variance channels are outliers
2. Split the input vector into outlier and regular subsets
3. Apply **separate random rotations** (with independent seeds) to each subset
4. Quantize each subset with its own codebook at its own bit width
5. Store norms separately for each subset
6. On decode: dequantize each subset with its own codebook, apply inverse rotation, reassemble

This avoids the approximation of using a single rotation with post-rotation variance detection, which was our earlier approach. The two-independent-rotations method is the theoretically optimal approach described in the paper.

## QJL Score Weight

The `compute_attention()` method defaults to `qjl_score_weight=1.0`, which produces the paper-correct **unbiased** inner product estimator (Theorem 2). Setting `qjl_score_weight < 1.0` trades bias for lower variance — this is a practical heuristic not present in the paper that can improve attention quality when the QJL variance is high relative to score differences.

## Backward Compatibility

The public API uses names like `PolarQuantCompressed` and `polarquant_encode/decode` for backward compatibility with existing code that imports these. Internally, these now implement scalar per-coordinate quantization (TurboQuant Algorithm 1), not recursive polar transform.
