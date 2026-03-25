# TurboQuant Implementation Pseudocode

> Extracted from: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
> Authors: Zandieh, Daliri, Hadian, Mirrokni (Google Research / NYU / Google DeepMind)
> Companion papers: [QJL](https://dl.acm.org/doi/10.1609/aaai.v39i24.34773), [PolarQuant](https://arxiv.org/abs/2502.02617)

---

## Table of Contents

1. [Constants & Setup](#1-constants--setup)
2. [polarquant_encode()](#2-polarquant_encodevector---compressed) — MSE-optimal quantization (Algorithm 1)
3. [polarquant_decode()](#3-polarquant_decodecompressed---vector) — MSE-optimal dequantization
4. [qjl_encode()](#4-qjl_encoderesidual---1bit) — 1-bit residual quantization
5. [qjl_correction()](#5-qjl_correctionquery-compressed_kv---corrected_score) — Unbiased inner product estimator
6. [turboquant_full_pipeline()](#6-turboquant_full_pipelinekv_vector---3bit_compressed) — Complete 3-bit pipeline (Algorithm 2)
7. [Attention Integration](#7-attention-integration) — How it plugs into transformer attention

---

## 1. Constants & Setup

### Dimensions (typical for llama.cpp)
```
d = head_dim             // e.g., 128 for Llama-2/3, 96 for Mistral
b_mse = 2               // bits per coordinate for MSE stage (PolarQuant)
b_qjl = 1               // bits per coordinate for QJL residual stage  
b_total = b_mse + b_qjl // = 3 bits per coordinate total
```

### Pre-initialization (done ONCE at model load time)

```
// === Random rotation matrix Π ∈ ℝ^{d×d} ===
// Must be a random orthogonal matrix. Generate via:
//   1. Sample G ∈ ℝ^{d×d} with G_ij ~ N(0,1)
//   2. QR decomposition: G = Q·R  
//   3. Π = Q (the orthogonal factor)
// IMPORTANT: Same Π is shared across ALL tokens in a layer/head.
// Use a deterministic seed per (layer, head) so encode/decode are consistent.
// In practice: use randomized Hadamard transform (RHT) for O(d log d) instead of O(d²):
//   Π·x = D·H·x  where H = Hadamard matrix, D = diagonal with random ±1 entries

seed_per_head = hash(layer_idx, head_idx)
rng = PRNG(seed_per_head)
D_signs = random_signs(d, rng)        // d-vector of ±1, for fast Hadamard rotation
// If using full rotation: Π = random_orthogonal_matrix(d, rng)

// === QJL random matrix S ∈ ℝ^{d×d} ===
// S_ij ~ N(0,1) i.i.d.
// Same S shared across all tokens in a layer/head.
// In practice: use structured random matrix (SRHT) for efficiency.
S = random_gaussian_matrix(d, d, rng)  // or generate on-the-fly from seed

// === Lloyd-Max codebook for Beta distribution ===
// Precomputed for dimension d and bit-width b_mse.
// After random rotation, each coordinate x_j of a unit-norm vector follows:
//   f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}
//   for x ∈ [-1, 1]
// In high dimensions (d ≥ 64), this converges to N(0, 1/d).
//
// For b_mse bits, we have K = 2^{b_mse} quantization levels.
// Lloyd-Max gives us:
//   centroids[0..K-1]: the reconstruction values (codebook entries)
//   boundaries[0..K]:  decision thresholds, boundaries[0] = -1, boundaries[K] = 1
//
// The Lloyd-Max algorithm iterates:
//   1. Given centroids c_0 < c_1 < ... < c_{K-1}:
//      boundaries[i] = (c_{i-1} + c_i) / 2  for i = 1..K-1
//      boundaries[0] = -1, boundaries[K] = 1
//   2. Given boundaries t_0 < t_1 < ... < t_K:
//      c_i = E[X | t_i ≤ X < t_{i+1}] = ∫_{t_i}^{t_{i+1}} x·f_X(x) dx / ∫_{t_i}^{t_{i+1}} f_X(x) dx
//   3. Repeat until convergence.
//
// PRECOMPUTED TABLE for d=128, using Gaussian approximation N(0, 1/128):
//   σ = 1/√d ≈ 0.0884
//
//   b=1 (K=2):  centroids = [-σ·√(2/π), +σ·√(2/π)] ≈ [-0.0705, +0.0705]
//               boundaries = [-1, 0, 1]
//
//   b=2 (K=4):  (standard Lloyd-Max for Gaussian, scaled by σ)
//               centroids  ≈ σ · [-1.510, -0.4528, +0.4528, +1.510]
//                          ≈ [-0.1335, -0.0400, +0.0400, +0.1335]
//               boundaries ≈ σ · [-∞, -0.9816, 0, +0.9816, +∞]
//                   clipped to [-1, -0.0868, 0, +0.0868, +1]
//
//   b=3 (K=8):  (standard Lloyd-Max for Gaussian, scaled by σ)
//               centroids  ≈ σ · [-2.152, -1.344, -0.7560, -0.2451,
//                                  +0.2451, +0.7560, +1.344, +2.152]
//               boundaries ≈ σ · [-∞, -1.748, -1.050, -0.5006, 0,
//                                  +0.5006, +1.050, +1.748, +∞]
//
// These are stored as constant arrays. For exact Beta distribution (small d),
// compute numerically. For d ≥ 64, Gaussian approximation is excellent.

CODEBOOK = precompute_lloyd_max_codebook(d, b_mse)
// CODEBOOK.centroids: float[K]    — reconstruction values
// CODEBOOK.boundaries: float[K+1] — decision boundaries
```

### Key Mathematical Facts

```
// Coordinate distribution after random rotation (Lemma 1):
//   x_j ~ f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}
//   For large d: x_j ≈ N(0, 1/d)
//   Variance: Var(x_j) = 1/d
//   Range: x_j ∈ [-1, 1]

// MSE distortion bound (Theorem 1):
//   E[‖x - Q_mse^{-1}(Q_mse(x))‖²] ≤ (√3 · π / 2) · (1/4^b)
//   For b=2: ≈ 0.117 (relative to ‖x‖² = 1)

// QJL variance bound (Lemma 4):
//   Var(⟨y, Q_qjl^{-1}(Q_qjl(x))⟩) ≤ (π / 2d) · ‖y‖²
//   for ‖x‖ = 1

// Inner product distortion bound (Theorem 2):
//   E[|⟨y,x⟩ - ⟨y, Q_prod^{-1}(Q_prod(x))⟩|²] ≤ (√3·π²·‖y‖²/d) · (1/4^b)
//   For b=3: ≈ 0.18·‖y‖²/d
```

---

## 2. polarquant_encode(vector) -> compressed

This is the MSE-optimal quantization stage (Algorithm 1 / "PolarQuant").

```pseudocode
function polarquant_encode(x: float[d]) -> PolarQuantCompressed:
    // INPUT:  x ∈ ℝ^d — a KV cache vector (key or value for one head)
    // OUTPUT: compressed representation using b_mse bits per coordinate
    
    // --- Step 1: Store and remove the norm ---
    // The algorithm assumes ‖x‖ = 1. For arbitrary vectors, store the norm
    // separately in full precision (float16 = 16 bits overhead per vector).
    norm = ‖x‖₂                        // = √(Σ x_i²)
    if norm < ε:                         // ε = 1e-10, avoid division by zero
        return {norm: 0, indices: zeros(d)}
    x_unit = x / norm                    // normalize to unit sphere S^{d-1}
    
    // --- Step 2: Random rotation ---
    // Apply the pre-generated random orthogonal rotation.
    // This transforms x_unit so each coordinate follows the Beta distribution.
    // Fast path: use randomized Hadamard transform (3 operations):
    //   1. x_rotated = D_signs ⊙ x_unit    (element-wise multiply by ±1)
    //   2. x_rotated = hadamard(x_rotated)  (fast Walsh-Hadamard, O(d log d))
    //   3. x_rotated = x_rotated / √d       (normalize)
    // Full path: x_rotated = Π · x_unit     (matrix-vector multiply, O(d²))
    
    x_rotated = apply_random_rotation(x_unit)    // x_rotated ∈ [-1, 1]^d
    
    // --- Step 3: Scalar quantization per coordinate ---
    // Each coordinate x_rotated[j] ∈ [-1, 1] follows the Beta distribution.
    // Quantize each independently using the precomputed Lloyd-Max codebook.
    // For K = 2^{b_mse} levels, each coordinate maps to an index in {0, 1, ..., K-1}.
    
    K = 2^b_mse                          // e.g., K=4 for b_mse=2
    indices = uint8[d]                    // or packed into b_mse bits per element
    
    for j = 0 to d-1:
        // Find which quantization bin x_rotated[j] falls into
        // Binary search over CODEBOOK.boundaries[0..K]
        idx = binary_search(CODEBOOK.boundaries, x_rotated[j])
        // idx ∈ {0, 1, ..., K-1} such that:
        //   CODEBOOK.boundaries[idx] ≤ x_rotated[j] < CODEBOOK.boundaries[idx+1]
        indices[j] = clamp(idx, 0, K-1)
    
    // --- Step 4: Pack into compressed representation ---
    // Total storage: d × b_mse bits for indices + 16 bits for norm
    // For b_mse=2, d=128: 256 bits (32 bytes) + 2 bytes norm = 34 bytes
    // vs. original: 128 × 16 = 2048 bits (256 bytes) at fp16
    
    return PolarQuantCompressed {
        norm:    float16(norm),           // 16 bits
        indices: pack_bits(indices, b_mse) // d × b_mse bits
    }
```

### Data Structure
```
struct PolarQuantCompressed:
    norm:    float16           // L2 norm of original vector (16 bits)
    indices: uint_packed       // d values, each b_mse bits wide
                               // Total: d × b_mse bits
    // Memory: 16 + d×b_mse bits per vector
    // For d=128, b_mse=2: 16 + 256 = 272 bits = 34 bytes
```

---

## 3. polarquant_decode(compressed) -> vector

```pseudocode
function polarquant_decode(c: PolarQuantCompressed) -> float[d]:
    // INPUT:  compressed PolarQuant representation
    // OUTPUT: reconstructed vector x̂ ∈ ℝ^d (approximation of original)
    
    if c.norm == 0:
        return zeros(d)
    
    // --- Step 1: Look up centroids from codebook ---
    indices = unpack_bits(c.indices, b_mse)    // d values in {0..K-1}
    
    x_rotated_hat = float[d]
    for j = 0 to d-1:
        x_rotated_hat[j] = CODEBOOK.centroids[indices[j]]
    
    // --- Step 2: Inverse random rotation ---
    // Π is orthogonal, so Π^{-1} = Π^T
    // Fast path: inverse Hadamard is same as forward (self-inverse), then undo D_signs
    //   1. x_hat = x_rotated_hat · √d
    //   2. x_hat = hadamard(x_hat)
    //   3. x_hat = D_signs ⊙ x_hat
    // Full path: x_hat = Π^T · x_rotated_hat
    
    x_unit_hat = apply_inverse_rotation(x_rotated_hat)
    
    // --- Step 3: Restore norm ---
    x_hat = float16(c.norm) * x_unit_hat
    
    return x_hat
    
    // PROPERTIES:
    // E[‖x - x_hat‖²] ≤ (√3·π/2) · ‖x‖² / 4^{b_mse}
    // For b_mse=2: E[‖x - x_hat‖²] ≤ 0.117 · ‖x‖²
    // NOTE: This is BIASED for inner products. Use TurboQuant for unbiased IP.
```

---

## 4. qjl_encode(residual) -> 1bit

The QJL (Quantized Johnson-Lindenstrauss) transform reduces the residual to 1 bit per coordinate.

```pseudocode
function qjl_encode(r: float[d]) -> QJLCompressed:
    // INPUT:  r ∈ ℝ^d — residual error vector from PolarQuant
    //         r = x_unit - polarquant_decode_unit(polarquant_encode(x_unit))
    //         where x_unit = x / ‖x‖
    // OUTPUT: 1-bit quantized representation (d bits total)
    
    // --- Step 1: Random projection ---
    // Multiply residual by random Gaussian matrix S ∈ ℝ^{d×d}
    // S_ij ~ N(0,1), same S used for encode and decode.
    // projected[i] = Σ_j S[i][j] · r[j]  for i = 0..d-1
    
    projected = S · r                    // matrix-vector multiply, O(d²)
                                          // projected ∈ ℝ^d
    
    // --- Step 2: Sign quantization (1-bit) ---
    // Keep only the sign of each projected coordinate.
    // sign(x) = +1 if x ≥ 0, -1 if x < 0
    
    signs = bit[d]
    for i = 0 to d-1:
        signs[i] = (projected[i] >= 0) ? 1 : 0    // pack as single bits
    
    // --- Step 3: Store residual norm ---
    // We need ‖r‖ for the QJL estimator to scale properly.
    // Since x was normalized to unit sphere before PolarQuant:
    //   ‖r‖² = ‖x_unit - x̂_unit‖² = MSE_distortion
    //   This is bounded and known from the codebook, so we can either:
    //   (a) Store ‖r‖ explicitly (16 bits), or
    //   (b) Compute it on-the-fly from the PolarQuant compressed data.
    // Option (b) is preferred to save storage.
    
    r_norm = ‖r‖₂
    
    return QJLCompressed {
        signs:  pack_bits(signs),         // d bits
        r_norm: float16(r_norm)           // 16 bits (optional, can recompute)
    }

    // STORAGE: d × 1 bits + 16 bits (norm)
    // For d=128: 128 + 16 = 144 bits = 18 bytes
```

### QJL Dequantization (for reference — NOT used directly in attention)

```pseudocode
function qjl_decode(c: QJLCompressed) -> float[d]:
    // This reconstructs the residual approximation.
    // Normally NOT called directly — use qjl_correction() instead.
    
    signs = unpack_signs(c.signs)         // d values in {-1, +1}
    
    // Dequantization formula (Definition 1 in paper):
    //   Q_qjl^{-1}(z) = (√(π/2) / d) · S^T · z
    
    r_hat = (sqrt(π/2) / d) * S^T · signs    // ∈ ℝ^d
    
    // Scale by residual norm (the QJL is defined for unit vectors)
    r_hat = c.r_norm * r_hat
    
    return r_hat
    
    // PROPERTIES (for unit-norm input r/‖r‖):
    // E[⟨y, r_hat⟩] = ⟨y, r⟩           (unbiased!)
    // Var(⟨y, r_hat⟩) ≤ (π/2d) · ‖y‖²  (variance bound)
```

---

## 5. qjl_correction(query, compressed_kv) -> corrected_score

This is the KEY function for attention computation. It computes an unbiased inner product estimate combining PolarQuant reconstruction with QJL correction.

```pseudocode
function qjl_correction(
    q:           float[d],              // query vector (FULL PRECISION)
    pq_compressed: PolarQuantCompressed, // PolarQuant-compressed key
    qjl_compressed: QJLCompressed        // QJL-compressed residual
) -> float:
    // INPUT:  full-precision query q, compressed key (PolarQuant + QJL parts)
    // OUTPUT: unbiased estimate of ⟨q, k_original⟩
    //
    // The true attention score is: score = ⟨q, k⟩
    // We decompose k = k̂ + r, where:
    //   k̂ = PolarQuant reconstruction
    //   r = k - k̂ = residual error
    // So: ⟨q, k⟩ = ⟨q, k̂⟩ + ⟨q, r⟩
    //
    // PolarQuant gives us k̂ (biased for IP).
    // QJL gives us an unbiased estimate of ⟨q, r⟩.
    // Combined: unbiased estimate of ⟨q, k⟩.
    
    // --- Part A: PolarQuant score (direct inner product with reconstruction) ---
    k_hat = polarquant_decode(pq_compressed)     // ℝ^d, the MSE reconstruction
    score_pq = dot(q, k_hat)                      // ⟨q, k̂⟩
    
    // --- Part B: QJL correction term ---
    // Instead of fully decoding the residual, compute the inner product directly.
    // This is more efficient and numerically stable.
    //
    // From Definition 1 (QJL):
    //   ⟨y, Q_qjl^{-1}(z)⟩ = (√(π/2) / d) · ⟨y, S^T · z⟩
    //                        = (√(π/2) / d) · ⟨S·y, z⟩
    //                        = (√(π/2) / d) · Σᵢ (S·y)ᵢ · zᵢ
    //
    // This means we can compute the QJL correction by:
    //   1. Project query through S:  q_proj = S · q
    //   2. Take sign-weighted sum:   Σᵢ q_proj[i] · signs[i]
    //   3. Scale by √(π/2) / d
    //
    // The residual r was stored as sign(S · (r/‖r‖)), so we also scale by ‖r‖.
    
    signs = unpack_signs(qjl_compressed.signs)    // d values in {-1, +1}
    
    // Project query through the SAME random matrix S used during encoding
    q_proj = S · q                                 // ℝ^d
    
    // Compute QJL inner product estimate for unit-norm residual
    qjl_ip = 0.0
    for i = 0 to d-1:
        qjl_ip += q_proj[i] * signs[i]            // signs[i] ∈ {-1, +1}
    
    qjl_ip = qjl_ip * sqrt(π/2) / d               // scale factor from QJL
    
    // Scale by residual norm to get estimate of ⟨q, r⟩
    score_qjl = qjl_ip * float(qjl_compressed.r_norm)
    
    // --- Combined unbiased score ---
    corrected_score = score_pq + score_qjl
    
    return corrected_score
    
    // PROPERTIES:
    // E[corrected_score] = ⟨q, k⟩        (UNBIASED!)
    // Var ≤ (√3·π²·‖q‖²/d) · (1/4^{b_total}) · ‖k‖²
    // For b_total=3, d=128: Var ≤ 0.18·‖q‖²·‖k‖²/128
```

### Optimized Batch Version (for CUDA kernel)

```pseudocode
function qjl_correction_batch(
    q:     float[d],                     // single query
    keys:  TurboQuantCompressed[n_seq],  // all compressed keys
) -> float[n_seq]:
    // Compute attention scores for all n_seq keys at once.
    // This is the hot path — must be highly optimized.
    
    scores = float[n_seq]
    
    // Pre-compute S·q ONCE (amortized across all keys)
    q_proj = S · q                        // O(d²), done once
    scale = sqrt(π/2) / d
    
    for t = 0 to n_seq-1:
        // PolarQuant part: decode and dot product
        k_hat = polarquant_decode(keys[t].pq)
        score_pq = dot(q, k_hat)
        
        // QJL part: sign-weighted sum (very fast, just d multiplies + adds)
        signs = unpack_signs(keys[t].qjl.signs)
        qjl_ip = dot(q_proj, signs)       // d multiply-adds
        score_qjl = qjl_ip * scale * keys[t].qjl.r_norm
        
        scores[t] = score_pq + score_qjl
    
    return scores
    
    // FLOPS per key: d (PolarQuant decode) + d (dot q,k_hat) + d (QJL dot) = 3d
    // vs. uncompressed: d (single dot product)
    // But memory bandwidth is 6x less → net speedup on memory-bound attention
```

---

## 6. turboquant_full_pipeline(kv_vector) -> 3bit_compressed

Complete encode/decode pipeline combining PolarQuant (2-bit) + QJL (1-bit) = 3-bit total.

```pseudocode
function turboquant_encode(x: float[d]) -> TurboQuantCompressed:
    // INPUT:  x ∈ ℝ^d — a key or value vector (one attention head)
    // OUTPUT: 3-bit compressed representation
    //
    // Algorithm 2 from the paper: "Inner-product Optimal TurboQuant"
    // Stage 1: Apply Q_mse with (b-1) bits → minimize ‖residual‖
    // Stage 2: Apply Q_qjl (1 bit) to the residual → unbiased IP
    
    // --- Stage 1: PolarQuant (b_mse = b_total - 1 = 2 bits) ---
    pq = polarquant_encode(x)            // 2 bits/coord + 16 bits norm
    
    // --- Compute residual ---
    x_hat = polarquant_decode(pq)         // MSE-optimal reconstruction
    residual = x - x_hat                  // r ∈ ℝ^d, the quantization error
    
    // --- Stage 2: QJL on residual (1 bit/coord) ---
    // Normalize residual to unit sphere for QJL (which assumes ‖input‖=1)
    r_norm = ‖residual‖₂
    if r_norm > ε:
        r_unit = residual / r_norm
    else:
        r_unit = zeros(d)
    
    qjl = qjl_encode(r_unit)             // 1 bit/coord + r_norm
    qjl.r_norm = float16(r_norm)          // store the residual norm
    
    return TurboQuantCompressed {
        pq:  pq,                          // PolarQuant part: d×2 + 16 bits
        qjl: qjl                          // QJL part: d×1 + 16 bits
    }
    
    // TOTAL STORAGE per vector:
    //   PolarQuant: d × b_mse bits + 16 bits (vector norm)
    //   QJL:        d × 1 bits    + 16 bits (residual norm)
    //   Total:      d × (b_mse + 1) + 32 bits
    //   For d=128, b_mse=2: 128×3 + 32 = 416 bits = 52 bytes
    //   vs. FP16: 128 × 16 = 2048 bits = 256 bytes
    //   Compression ratio: 2048 / 416 ≈ 4.9x
    //   Effective bits per value: 416/128 = 3.25 (3 + small overhead)


function turboquant_decode(c: TurboQuantCompressed) -> float[d]:
    // Full reconstruction (for debugging / non-attention uses).
    // For attention, use qjl_correction() instead — it's more efficient.
    
    k_hat = polarquant_decode(c.pq)       // MSE reconstruction
    r_hat = qjl_decode(c.qjl)             // QJL residual reconstruction
    
    return k_hat + r_hat
```

### Data Structures

```
struct TurboQuantCompressed:
    pq:  PolarQuantCompressed    // MSE stage (2 bits/coord + norm)
    qjl: QJLCompressed           // Residual stage (1 bit/coord + norm)

// Memory layout for d=128, b_mse=2 (PACKED):
//
// Offset  Size    Field
// 0       2B      pq.norm (float16)
// 2       32B     pq.indices (128 × 2 bits = 256 bits = 32 bytes)
// 34      2B      qjl.r_norm (float16)
// 36      16B     qjl.signs (128 × 1 bit = 128 bits = 16 bytes)
// ─────────────────────────────
// Total:  52 bytes per vector (vs. 256 bytes at fp16)
//
// Alternative: interleave PQ indices and QJL signs for better cache locality:
// Per coordinate: 2 bits (PQ) + 1 bit (QJL) = 3 bits
// Pack 128 × 3 = 384 bits = 48 bytes + 4 bytes norms = 52 bytes total
```

---

## 7. Attention Integration

How TurboQuant plugs into the transformer attention computation in llama.cpp.

### KV Cache Write (on each new token)

```pseudocode
function kv_cache_store(layer, head, pos, k_vec, v_vec):
    // Called after computing K and V projections for a new token.
    // k_vec, v_vec ∈ ℝ^{head_dim} in float16
    
    // Compress and store
    kv_cache[layer][head][pos].key   = turboquant_encode(k_vec)
    kv_cache[layer][head][pos].value = turboquant_encode(v_vec)
    
    // Storage: 52 bytes per key + 52 bytes per value = 104 bytes per position
    // vs. FP16: 256 + 256 = 512 bytes per position
    // Savings: ~4.9x
```

### Attention Score Computation (the hot path)

```pseudocode
function compute_attention(layer, head, q_vec, seq_len):
    // q_vec ∈ ℝ^{head_dim} — current query (FULL PRECISION, not quantized!)
    // Returns: attention output ∈ ℝ^{head_dim}
    
    // --- Step 1: Compute attention scores ---
    scores = float[seq_len]
    
    // Pre-project query through QJL matrix (done ONCE)
    q_proj = S[layer][head] · q_vec       // for QJL correction
    qjl_scale = sqrt(π/2) / d
    
    for t = 0 to seq_len-1:
        kc = kv_cache[layer][head][t].key
        
        // PolarQuant inner product
        k_hat = polarquant_decode(kc.pq)
        score_pq = dot(q_vec, k_hat)
        
        // QJL correction
        signs = unpack_signs(kc.qjl.signs)
        score_qjl = dot(q_proj, signs) * qjl_scale * kc.qjl.r_norm
        
        scores[t] = (score_pq + score_qjl) / sqrt(d)  // scaled dot-product attention
    
    // --- Step 2: Softmax ---
    attn_weights = softmax(scores)        // float[seq_len]
    
    // --- Step 3: Weighted sum of values ---
    // Values also compressed with TurboQuant.
    // Option A: Decode each value and compute weighted sum (simpler).
    // Option B: Use QJL correction for values too (mathematically equivalent
    //           since we need the actual vector, not just inner product).
    // For values, we typically just decode since we need the full vector.
    
    output = zeros(d)
    for t = 0 to seq_len-1:
        if attn_weights[t] > threshold:   // skip near-zero weights
            vc = kv_cache[layer][head][t].value
            v_hat = turboquant_decode(vc)  // full decode for values
            output += attn_weights[t] * v_hat
    
    return output
```

### CUDA Kernel Sketch

```pseudocode
// GPU kernel for batched attention with TurboQuant keys
__global__ void turboquant_attention_kernel(
    float* q,                    // [n_heads, head_dim] queries
    TurboQuantPacked* k_cache,   // [n_heads, seq_len] packed keys
    TurboQuantPacked* v_cache,   // [n_heads, seq_len] packed values
    float* output,               // [n_heads, head_dim] output
    int seq_len, int head_dim
):
    head = blockIdx.x
    
    // Load query into shared memory
    __shared__ float q_local[MAX_HEAD_DIM]
    __shared__ float q_proj[MAX_HEAD_DIM]    // S·q precomputed
    
    // Cooperative load of query
    q_local[threadIdx.x] = q[head * head_dim + threadIdx.x]
    __syncthreads()
    
    // Precompute S·q (use structured matrix for speed)
    // Each thread computes one element of q_proj
    q_proj[threadIdx.x] = dot_row(S_matrix, threadIdx.x, q_local, head_dim)
    __syncthreads()
    
    // Each thread handles a chunk of the sequence
    float scale = sqrtf(M_PI / 2.0f) / head_dim
    
    for t = threadIdx.x; t < seq_len; t += blockDim.x:
        // Unpack PolarQuant indices (2 bits each)
        // Compute ⟨q, codebook[indices]⟩ using lookup table
        float score_pq = polarquant_dot_product(q_local, k_cache[head][t].pq)
        
        // QJL correction: dot(q_proj, signs) — very fast bit operations
        // signs are packed as bits, use popcount tricks
        float score_qjl = qjl_dot_product(q_proj, k_cache[head][t].qjl.signs)
        score_qjl *= scale * k_cache[head][t].qjl.r_norm
        
        scores[t] = (score_pq + score_qjl) * rsqrtf(head_dim)
    
    // ... softmax and value aggregation follow ...
```

---

## Appendix A: Lloyd-Max Codebook Computation

```pseudocode
function precompute_lloyd_max_codebook(d: int, b: int) -> Codebook:
    // Computes the optimal scalar quantizer for the coordinate distribution
    // of a random point on S^{d-1}.
    //
    // For large d (≥ 64): use Gaussian approximation N(0, 1/d)
    // For small d: use exact Beta distribution numerically
    
    K = 2^b                              // number of levels
    σ = 1.0 / sqrt(d)                    // std dev of coordinate distribution
    
    // Initialize centroids uniformly in [-3σ, 3σ]
    centroids = linspace(-3*σ + σ/(2*K), 3*σ - σ/(2*K), K)
    boundaries = float[K+1]
    
    // Lloyd-Max iteration
    for iter = 1 to MAX_ITER:            // typically converges in 20-50 iterations
        // Step 1: Update boundaries (midpoints between centroids)
        boundaries[0] = -1.0             // hard lower bound (unit sphere)
        boundaries[K] = 1.0              // hard upper bound
        for i = 1 to K-1:
            boundaries[i] = (centroids[i-1] + centroids[i]) / 2.0
        
        // Step 2: Update centroids (conditional expectations)
        for i = 0 to K-1:
            lo = boundaries[i]
            hi = boundaries[i+1]
            
            // c_i = E[X | lo ≤ X < hi]
            //      = ∫_{lo}^{hi} x · f_X(x) dx / ∫_{lo}^{hi} f_X(x) dx
            //
            // For Gaussian approximation:
            //   f_X(x) = (1/(σ√(2π))) · exp(-x²/(2σ²))
            //   numerator   = σ² · (f_X(lo) - f_X(hi))     [standard result]
            //   denominator = Φ(hi/σ) - Φ(lo/σ)            [Gaussian CDF]
            
            num = σ * (gaussian_pdf(lo/σ) - gaussian_pdf(hi/σ))
            den = gaussian_cdf(hi/σ) - gaussian_cdf(lo/σ)
            
            if den > ε:
                centroids[i] = num / den * σ
                // Correction: centroids[i] = σ² · (φ(lo/σ) - φ(hi/σ)) / (Φ(hi/σ) - Φ(lo/σ))
                // where φ = Gaussian PDF, Φ = Gaussian CDF
            else:
                centroids[i] = (lo + hi) / 2.0
        
        // Check convergence
        if max change in centroids < 1e-10:
            break
    
    return Codebook { centroids, boundaries }

// REFERENCE VALUES for common configurations:
//
// d=128, b=2 (K=4), σ=0.0884:
//   centroids  ≈ [-0.1335, -0.0400, +0.0400, +0.1335]
//   boundaries ≈ [-1.0, -0.0868, 0.0, +0.0868, +1.0]
//   MSE per coord ≈ 7.19e-4 → total MSE ≈ 0.092 (< 0.117 bound)
//
// d=128, b=3 (K=8), σ=0.0884:
//   centroids  ≈ [-0.190, -0.119, -0.0668, -0.0217,
//                  +0.0217, +0.0668, +0.119, +0.190]
//   boundaries ≈ [-1.0, -0.155, -0.0928, -0.0442, 0.0,
//                  +0.0442, +0.0928, +0.155, +1.0]
//   MSE per coord ≈ 1.78e-4 → total MSE ≈ 0.023 (< 0.03 bound)
```

---

## Appendix B: Randomized Hadamard Transform (Fast Rotation)

```pseudocode
// O(d log d) alternative to O(d²) full rotation matrix multiplication.
// Used in practice for PolarQuant.

function apply_random_rotation(x: float[d]) -> float[d]:
    // Randomized Hadamard Transform (RHT)
    // Π·x = (1/√d) · H · D · x
    // where D = diag(±1) random signs, H = Walsh-Hadamard matrix
    
    // Step 1: Random sign flip
    y = D_signs ⊙ x                     // element-wise multiply by ±1
    
    // Step 2: Fast Walsh-Hadamard Transform (in-place, O(d log d))
    // Requires d = power of 2 (true for typical head_dim: 64, 128, 256)
    fwht_inplace(y, d)
    
    // Step 3: Normalize
    y = y / sqrt(d)
    
    return y

function apply_inverse_rotation(x: float[d]) -> float[d]:
    // Inverse: Π^T·x = D^T · H^T · (1/√d) · x = D · (1/√d) · H · x
    // Since H is symmetric and D is its own inverse:
    
    y = x * sqrt(d)                      // undo normalization first? No:
    // Actually: Π = (1/√d)·H·D, so Π^T = (1/√d)·D^T·H^T = (1/√d)·D·H
    
    y = fwht_inplace(x, d)              // Hadamard
    y = y / sqrt(d)                      // normalize (H·H = d·I)
    // Wait — FWHT is self-inverse up to scaling: H·(H·x) = d·x
    // So if forward was: y = (1/√d)·H·(D·x)
    // Then inverse is:   x = D · (1/√d) · H · y
    //                      = D · (1/√d) · H · ((1/√d)·H·D·x)
    //                      = D · (1/d) · d · D · x = x  ✓
    
    y = x.copy()
    fwht_inplace(y, d)                   // H · y
    y = y / sqrt(d)                      // (1/√d) · H · y
    y = D_signs ⊙ y                     // D · result
    
    return y

function fwht_inplace(x: float[d], d: int):
    // Fast Walsh-Hadamard Transform, iterative, in-place
    // Complexity: O(d log d)
    h = 1
    while h < d:
        for i = 0 to d-1 step 2*h:
            for j = i to i+h-1:
                a = x[j]
                b = x[j + h]
                x[j]     = a + b
                x[j + h] = a - b
        h *= 2
```

---

## Appendix C: Summary of Bit Budget

| Component | Bits per coordinate | Bits per vector (d=128) | Purpose |
|-----------|-------------------|------------------------|---------|
| PolarQuant indices | b_mse (2) | 256 | MSE-optimal reconstruction |
| PolarQuant norm | — | 16 | Vector magnitude |
| QJL signs | 1 | 128 | Unbiased IP correction |
| QJL residual norm | — | 16 | Residual magnitude |
| **Total** | **3** | **416** | **3.25 effective bits/value** |
| FP16 baseline | 16 | 2048 | Full precision |
| **Compression ratio** | | **4.9×** | |

### Distortion Guarantees (‖x‖=1, d=128)

| Metric | b=2 (PolarQuant only) | b=3 (TurboQuant) | Lower bound |
|--------|----------------------|-------------------|-------------|
| MSE: E[‖x-x̂‖²] | ≤ 0.117 | ≤ 0.030 | ≥ 1/4^b |
| IP: E[\|⟨y,x⟩-⟨y,x̂⟩\|²] | biased! | ≤ 0.18·‖y‖²/d | ≥ ‖y‖²/(d·4^b) |
| Unbiased? | ❌ No | ✅ Yes | — |

---

## Appendix D: Implementation Notes for llama.cpp

### File Mapping
```
ggml-common.h          — Add TurboQuant type definitions (GGML_TYPE_TQ3_0)
ggml-quants.c          — Scalar encode/decode functions
ggml-cuda/fattn*.cu    — CUDA attention kernels with TQ support
llama-kv-cache.cpp     — KV cache alloc/store/retrieve with TQ
llama.cpp              — CLI flag --kv-cache-type turboquant
```

### Performance Considerations
1. **Codebook lookup**: Store codebook in GPU constant memory (fits in 64 bytes for K=4)
2. **S matrix**: Generate from seed using cuRAND, don't store the full d×d matrix
3. **Hadamard**: Use CUDA cooperative groups for parallel butterfly operations
4. **QJL dot product**: Pack signs into uint32/uint64, use `__popc()` for Hamming-style computation
5. **Batch amortization**: Precompute `S·q` once per query across all seq_len keys

### Quantization Timing (estimated)
- Encode: ~2μs per vector (d=128) — dominated by Hadamard + codebook lookup
- Decode: ~1.5μs per vector
- Attention: memory-bandwidth bound → TurboQuant reads 52B vs 256B → ~4.9× speedup
