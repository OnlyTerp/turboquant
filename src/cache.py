"""
TurboQuant KV Cache Compression — Pure PyTorch Implementation

Implements the TurboQuant algorithm (Algorithm 2) for near-optimal vector
quantization of transformer KV caches, combining:
  - PolarQuant (2-bit MSE-optimal scalar quantization)
  - QJL (1-bit residual correction for unbiased inner products)

Total: 3 bits per coordinate → ~4.9× compression vs FP16.

Reference: https://arxiv.org/abs/2504.19874
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B_MSE = 2          # bits per coordinate for PolarQuant stage
B_QJL = 1          # bits per coordinate for QJL residual stage
B_TOTAL = B_MSE + B_QJL  # = 3 bits total per coordinate
EPS = 1e-10        # numerical stability threshold


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform (FWHT)
# ---------------------------------------------------------------------------

def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform.

    Self-inverse up to scaling: FWHT(FWHT(x)) = d * x.
    Preserves norms up to scaling: ‖FWHT(x)‖² = d · ‖x‖².
    Complexity: O(d log d).

    Args:
        x: [..., d] where d is a power of 2.

    Returns:
        Transformed tensor, same shape as input.
    """
    d = x.shape[-1]
    y = x.clone()
    h = 1
    while h < d:
        y_view = y.reshape(*y.shape[:-1], -1, 2 * h)
        a = y_view[..., :h].clone()
        b = y_view[..., h:].clone()
        y_view[..., :h] = a + b
        y_view[..., h:] = a - b
        y = y_view.reshape(*y.shape)
        h *= 2
    return y


def fwht_inplace(x: torch.Tensor) -> None:
    """In-place Fast Walsh-Hadamard Transform."""
    d = x.shape[-1]
    h = 1
    while h < d:
        y_view = x.reshape(*x.shape[:-1], -1, 2 * h)
        a = y_view[..., :h].clone()
        b = y_view[..., h:].clone()
        y_view[..., :h] = a + b
        y_view[..., h:] = a - b
        h *= 2


# ---------------------------------------------------------------------------
# Randomized Hadamard Transform (rotation for PolarQuant)
# ---------------------------------------------------------------------------

def _generate_signs(d: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate deterministic random ±1 signs from a seed."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randint(0, 2, (d,), generator=g, device=device, dtype=torch.float32) * 2 - 1


class RandomHadamardRotation:
    """Randomized Hadamard Transform: Π·x = (1/√d) · H · (D_signs ⊙ x).

    Inverse: Π^T · y = D_signs ⊙ ((1/√d) · H · y).
    """

    def __init__(self, d: int, seed: int, device: torch.device = torch.device("cpu")):
        self.d = d
        self.seed = seed
        self.sqrt_d = math.sqrt(d)
        self.signs = _generate_signs(d, seed, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation: Π · x."""
        y = x * self.signs
        fwht_inplace(y)
        y = y / self.sqrt_d
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation: Π^T · y = D ⊙ ((1/√d) · H · y)."""
        z = y.clone()
        fwht_inplace(z)
        z = z / self.sqrt_d
        z = z * self.signs
        return z


# ---------------------------------------------------------------------------
# Gaussian helpers for Lloyd-Max
# ---------------------------------------------------------------------------

def _gaussian_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard Gaussian PDF φ(x) = (1/√(2π)) · exp(-x²/2)."""
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard Gaussian CDF Φ(x)."""
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Lloyd-Max Codebook
# ---------------------------------------------------------------------------

@dataclass
class Codebook:
    """Lloyd-Max scalar quantizer codebook."""
    centroids: torch.Tensor    # [K] reconstruction values
    boundaries: torch.Tensor   # [K+1] decision boundaries
    d: int
    b: int
    K: int
    sigma: float

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map continuous values to codebook indices."""
        idx = torch.searchsorted(self.boundaries, x, right=False) - 1
        return idx.clamp(0, self.K - 1).long()

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        """Map codebook indices to centroid values."""
        return self.centroids[idx.long()]


def compute_lloyd_max_codebook(
    d: int,
    b: int,
    max_iter: int = 500,
    tol: float = 1e-12,
    device: torch.device = torch.device("cpu"),
) -> Codebook:
    """Compute the Lloyd-Max scalar quantizer for N(0, 1/d).

    After random rotation, each coordinate of a unit vector follows
    approximately N(0, σ²) where σ = 1/√d.
    """
    K = 2 ** b
    sigma = 1.0 / math.sqrt(d)

    lo_init = -3 * sigma + sigma / (2 * K)
    hi_init = 3 * sigma - sigma / (2 * K)
    centroids = torch.linspace(lo_init, hi_init, K, device=device)
    boundaries = torch.zeros(K + 1, device=device)

    for _ in range(max_iter):
        boundaries[0] = -1.0
        boundaries[K] = 1.0
        for i in range(1, K):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

        old_centroids = centroids.clone()
        for i in range(K):
            lo = boundaries[i]
            hi = boundaries[i + 1]
            lo_s = lo / sigma
            hi_s = hi / sigma
            # E[X | lo ≤ X < hi] = σ · (φ(lo/σ) - φ(hi/σ)) / (Φ(hi/σ) - Φ(lo/σ))
            num = sigma * (_gaussian_pdf(lo_s) - _gaussian_pdf(hi_s))
            den = _gaussian_cdf(hi_s) - _gaussian_cdf(lo_s)
            if den.item() > EPS:
                centroids[i] = num / den
            else:
                centroids[i] = (lo + hi) / 2.0

        if (centroids - old_centroids).abs().max().item() < tol:
            break

    return Codebook(centroids=centroids, boundaries=boundaries, d=d, b=b, K=K, sigma=sigma)


# ---------------------------------------------------------------------------
# QJL Random Matrix
# ---------------------------------------------------------------------------

def generate_qjl_matrix(d: int, seed: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Generate the QJL random Gaussian matrix S ∈ ℝ^{d×d}."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(d, d, generator=g, device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# PolarQuant: 2-bit MSE-optimal quantization
# ---------------------------------------------------------------------------

@dataclass
class PolarQuantCompressed:
    """Compressed representation from PolarQuant (2-bit per coord + norm)."""
    norm: torch.Tensor            # [batch] L2 norms
    indices: torch.Tensor         # [batch, d] uint8 indices in {0..K-1}
    codebook: Codebook
    rotation: RandomHadamardRotation

    @property
    def d(self) -> int:
        return self.indices.shape[-1]


def polarquant_encode(x: torch.Tensor, codebook: Codebook, rotation: RandomHadamardRotation) -> PolarQuantCompressed:
    """PolarQuant encode: MSE-optimal 2-bit quantization (Algorithm 1)."""
    if x.dim() == 1:
        x = x.unsqueeze(0)

    norm = x.norm(dim=-1)
    safe_norm = norm.clamp(min=EPS)
    x_unit = x / safe_norm.unsqueeze(-1)

    zero_mask = norm < EPS

    x_rotated = rotation.forward(x_unit)
    x_rotated = x_rotated.clamp(-1.0, 1.0)

    indices = codebook.quantize(x_rotated)

    if zero_mask.any():
        indices[zero_mask] = 0

    return PolarQuantCompressed(norm=norm, indices=indices, codebook=codebook, rotation=rotation)


def polarquant_decode(c: PolarQuantCompressed) -> torch.Tensor:
    """PolarQuant decode: reconstruct vectors from compressed form."""
    x_rotated_hat = c.codebook.centroids[c.indices]
    x_unit_hat = c.rotation.inverse(x_rotated_hat)
    x_hat = c.norm.unsqueeze(-1) * x_unit_hat

    zero_mask = c.norm < EPS
    if zero_mask.any():
        x_hat[zero_mask] = 0.0

    return x_hat


# ---------------------------------------------------------------------------
# QJL: 1-bit residual quantization
# ---------------------------------------------------------------------------

@dataclass
class QJLCompressed:
    """Compressed representation from QJL (1-bit per coord + residual norm)."""
    signs: torch.Tensor       # [batch, d] in {0, 1}
    r_norm: torch.Tensor      # [batch] residual norm
    S: torch.Tensor           # [d, d] random Gaussian matrix

    @property
    def d(self) -> int:
        return self.signs.shape[-1]


def qjl_encode(residual: torch.Tensor, S: torch.Tensor) -> QJLCompressed:
    """QJL encode: 1-bit sign quantization of projected residual."""
    if residual.dim() == 1:
        residual = residual.unsqueeze(0)

    r_norm = residual.norm(dim=-1)
    safe_norm = r_norm.clamp(min=EPS)
    r_unit = residual / safe_norm.unsqueeze(-1)

    projected = r_unit @ S.T  # [batch, d]
    signs = (projected >= 0).long()

    return QJLCompressed(signs=signs, r_norm=r_norm, S=S)


# ---------------------------------------------------------------------------
# TurboQuant: Complete 3-bit pipeline
# ---------------------------------------------------------------------------

@dataclass
class TurboQuantCompressed:
    """Complete TurboQuant compressed representation (3-bit per coord)."""
    pq: PolarQuantCompressed
    qjl: QJLCompressed

    @property
    def d(self) -> int:
        return self.pq.d


class TurboQuantConfig:
    """Configuration for a TurboQuant cache."""

    def __init__(self, d: int = 128, b_mse: int = B_MSE, device: torch.device = torch.device("cpu")):
        self.d = d
        self.b_mse = b_mse
        self.device = device
        self.codebook = compute_lloyd_max_codebook(d, b_mse, device=device)

    def make_rotation(self, layer_idx: int, head_idx: int) -> RandomHadamardRotation:
        # Deterministic seed independent of PYTHONHASHSEED
        seed = ((layer_idx * 1000003) ^ (head_idx * 999979) ^ 0xA5A5A5A5) & 0xFFFFFFFF
        return RandomHadamardRotation(self.d, seed, self.device)

    def make_qjl_matrix(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        # Deterministic seed independent of PYTHONHASHSEED
        seed = ((layer_idx * 1000003) ^ (head_idx * 999979) ^ 0x5A5A5A5A) & 0xFFFFFFFF
        return generate_qjl_matrix(self.d, seed, self.device)


def turboquant_encode_internal(
    x: torch.Tensor,
    codebook: Codebook,
    rotation: RandomHadamardRotation,
    S: torch.Tensor,
) -> TurboQuantCompressed:
    """Full TurboQuant encode: PolarQuant + QJL (Algorithm 2)."""
    if x.dim() == 1:
        x = x.unsqueeze(0)

    pq = polarquant_encode(x, codebook, rotation)
    x_hat = polarquant_decode(pq)
    residual = x - x_hat
    qjl = qjl_encode(residual, S)

    return TurboQuantCompressed(pq=pq, qjl=qjl)


def turboquant_decode_single(c: TurboQuantCompressed) -> torch.Tensor:
    """Full TurboQuant decode: PQ reconstruction + QJL residual estimate."""
    k_hat = polarquant_decode(c.pq)  # [1, d]

    signs_f = c.qjl.signs.float() * 2 - 1  # {-1, +1}
    d = c.d
    scale = math.sqrt(math.pi / 2) / d
    r_hat = (signs_f @ c.qjl.S) * scale  # [1, d]
    r_hat = r_hat * c.qjl.r_norm.unsqueeze(-1)

    return k_hat + r_hat


# ---------------------------------------------------------------------------
# TurboQuant Cache
# ---------------------------------------------------------------------------

class TurboQuantCache:
    """TurboQuant-compressed KV cache for transformer attention."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d: int = 128,
        b_mse: int = B_MSE,
        device: torch.device = torch.device("cpu"),
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d = d
        self.device = device
        self.config = TurboQuantConfig(d, b_mse, device=device)

        self.rotations: List[List[RandomHadamardRotation]] = []
        self.qjl_matrices: List[List[torch.Tensor]] = []
        for l in range(n_layers):
            self.rotations.append([])
            self.qjl_matrices.append([])
            for h in range(n_heads):
                self.rotations[l].append(self.config.make_rotation(l, h))
                self.qjl_matrices[l].append(self.config.make_qjl_matrix(l, h))

        # cache[layer][head] = list of (key_compressed, value_compressed)
        self.cache: List[List[List[Tuple[TurboQuantCompressed, TurboQuantCompressed]]]] = []
        for l in range(n_layers):
            self.cache.append([])
            for h in range(n_heads):
                self.cache[l].append([])

    @property
    def seq_len(self) -> int:
        if self.n_layers == 0 or self.n_heads == 0:
            return 0
        return len(self.cache[0][0])

    def store(self, layer_idx: int, head_idx: int, k_vec: torch.Tensor, v_vec: torch.Tensor):
        """Encode and store a key-value pair."""
        rotation = self.rotations[layer_idx][head_idx]
        S = self.qjl_matrices[layer_idx][head_idx]
        k_c = turboquant_encode_internal(k_vec, self.config.codebook, rotation, S)
        v_c = turboquant_encode_internal(v_vec, self.config.codebook, rotation, S)
        self.cache[layer_idx][head_idx].append((k_c, v_c))

    def store_batch(self, layer_idx: int, head_idx: int, k_vecs: torch.Tensor, v_vecs: torch.Tensor):
        """Encode and store a batch of key-value pairs."""
        rotation = self.rotations[layer_idx][head_idx]
        S = self.qjl_matrices[layer_idx][head_idx]
        k_all = turboquant_encode_internal(k_vecs, self.config.codebook, rotation, S)
        v_all = turboquant_encode_internal(v_vecs, self.config.codebook, rotation, S)

        for i in range(k_vecs.shape[0]):
            k_single = TurboQuantCompressed(
                pq=PolarQuantCompressed(
                    norm=k_all.pq.norm[i:i+1], indices=k_all.pq.indices[i:i+1],
                    codebook=k_all.pq.codebook, rotation=k_all.pq.rotation,
                ),
                qjl=QJLCompressed(
                    signs=k_all.qjl.signs[i:i+1], r_norm=k_all.qjl.r_norm[i:i+1], S=S,
                ),
            )
            v_single = TurboQuantCompressed(
                pq=PolarQuantCompressed(
                    norm=v_all.pq.norm[i:i+1], indices=v_all.pq.indices[i:i+1],
                    codebook=v_all.pq.codebook, rotation=v_all.pq.rotation,
                ),
                qjl=QJLCompressed(
                    signs=v_all.qjl.signs[i:i+1], r_norm=v_all.qjl.r_norm[i:i+1], S=S,
                ),
            )
            self.cache[layer_idx][head_idx].append((k_single, v_single))

    def compute_attention(
        self, layer_idx: int, head_idx: int, q_vec: torch.Tensor, causal: bool = True,
        qjl_score_weight: float = 0.3,
    ) -> torch.Tensor:
        """Compute attention output using compressed KV cache.

        Uses PolarQuant decode for key scores with a damped QJL correction term,
        and PolarQuant-only decode for values.

        The QJL correction adds an unbiased estimate of the residual inner product,
        but a single-sample estimate has high variance. A weight < 1.0 reduces
        variance at the cost of slight bias, improving attention quality in practice.
        Set qjl_score_weight=0.0 to use PolarQuant-only scoring.
        Set qjl_score_weight=1.0 for the full unbiased QJL correction.

        Values are decoded with PolarQuant-only (not QJL), because a single-sample
        QJL residual estimate for vector reconstruction has too high a variance to
        improve MSE over PolarQuant alone.
        """
        d = self.d
        seq_len = len(self.cache[layer_idx][head_idx])
        if seq_len == 0:
            return torch.zeros(d, device=self.device)

        q_vec = q_vec.float()
        S = self.qjl_matrices[layer_idx][head_idx]
        qjl_scale = math.sqrt(math.pi / 2) / d

        # Pre-project query through S ONCE (amortized over all keys)
        # q_proj[i] = (S @ q)[i] = S[i,:] . q  — needed for QJL correction
        q_proj = S @ q_vec  # [d]

        # --- Batch-decode all PQ keys and compute scores ---
        # Collecting indices/norms for vectorized decode is faster than per-token loop
        pq_norms = torch.stack([
            self.cache[layer_idx][head_idx][t][0].pq.norm.squeeze(0)
            for t in range(seq_len)
        ])  # [seq_len]
        pq_indices = torch.cat([
            self.cache[layer_idx][head_idx][t][0].pq.indices
            for t in range(seq_len)
        ], dim=0)  # [seq_len, d]

        rotation = self.rotations[layer_idx][head_idx]
        # Batch PolarQuant decode — builds [seq_len, d] key estimates
        pq_batch = PolarQuantCompressed(
            norm=pq_norms,
            indices=pq_indices,
            codebook=self.config.codebook,
            rotation=rotation,
        )
        k_hat_all = polarquant_decode(pq_batch)  # [seq_len, d]
        score_pq_all = (k_hat_all @ q_vec) / math.sqrt(d)  # [seq_len]

        if qjl_score_weight > 0.0:
            # QJL correction: for each key t, correction = dot(S·q, sign(S·r)) * scale * r_norm
            # Batch compute: signs_pm [seq_len, d], q_proj [d]
            signs_pm_all = torch.cat([
                self.cache[layer_idx][head_idx][t][0].qjl.signs
                for t in range(seq_len)
            ], dim=0).float() * 2 - 1  # [seq_len, d]  values ∈ {-1, +1}

            r_norms_all = torch.stack([
                self.cache[layer_idx][head_idx][t][0].qjl.r_norm.squeeze(0)
                for t in range(seq_len)
            ]).float()  # [seq_len]

            # score_qjl[t] = dot(q_proj, signs_pm_all[t]) * qjl_scale * r_norm[t]
            qjl_ips = (signs_pm_all @ q_proj)  # [seq_len]
            score_qjl_all = qjl_ips * qjl_scale * r_norms_all / math.sqrt(d)  # [seq_len]

            scores = score_pq_all + qjl_score_weight * score_qjl_all
        else:
            scores = score_pq_all

        # Softmax
        attn_weights = F.softmax(scores, dim=0)  # [seq_len]

        # --- Batch-decode all PQ values ---
        v_pq_norms = torch.stack([
            self.cache[layer_idx][head_idx][t][1].pq.norm.squeeze(0)
            for t in range(seq_len)
        ])
        v_pq_indices = torch.cat([
            self.cache[layer_idx][head_idx][t][1].pq.indices
            for t in range(seq_len)
        ], dim=0)

        v_pq_batch = PolarQuantCompressed(
            norm=v_pq_norms,
            indices=v_pq_indices,
            codebook=self.config.codebook,
            rotation=rotation,
        )
        v_hat_all = polarquant_decode(v_pq_batch)  # [seq_len, d]
        # Note: We use PolarQuant-only for values. A single-sample QJL residual
        # reconstruction has too high a variance to improve over PQ alone.
        # The QJL data is stored for potential multi-sample averaging in future work.

        # Weighted sum: [seq_len] x [seq_len, d] → [d]
        output = (attn_weights.unsqueeze(-1) * v_hat_all.float()).sum(0)

        return output


# ---------------------------------------------------------------------------
# Utility: Compression ratio analysis
# ---------------------------------------------------------------------------

def compression_ratio_fp16(d: int, b_mse: int = B_MSE) -> float:
    """Compute compression ratio vs FP16."""
    fp16_bits = d * 16
    tq_bits = d * b_mse + 16 + d * 1 + 16  # PQ + norm + QJL signs + r_norm
    return fp16_bits / tq_bits


def memory_bytes_per_vector(d: int, b_mse: int = B_MSE) -> Tuple[int, int]:
    """Returns (tq_bytes, fp16_bytes) per vector."""
    tq_bits = d * b_mse + 16 + d * 1 + 16
    tq_bytes = (tq_bits + 7) // 8
    fp16_bytes = d * 2
    return tq_bytes, fp16_bytes
